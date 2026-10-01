#!/usr/bin/env python3
"""PyGuide AI local app server with a restricted, short-lived Python runner."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
MAX_CODE_CHARS = 8_000
MAX_AST_NODES = 1_200
MAX_OUTPUT_CHARS = 20_000
SAFE_MODULES = {
    "math": {"sqrt", "floor", "ceil", "factorial", "gcd", "lcm", "pi", "e", "tau", "sin", "cos", "tan", "radians", "degrees", "log", "log2", "log10", "prod", "isclose", "isfinite", "inf"},
    "statistics": {"mean", "median", "mode", "stdev", "pstdev", "variance", "pvariance"},
}
SAFE_BUILTINS = {
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "iter", "len", "list", "map", "max", "min", "next", "print", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "pow", "divmod", "ord", "chr",
    "Exception", "ArithmeticError", "AssertionError", "IndexError", "KeyError", "NameError", "NotImplementedError", "RuntimeError", "TypeError", "ValueError", "ZeroDivisionError",
}
SAFE_METHODS = {
    "append", "clear", "copy", "count", "extend", "get", "index", "insert", "items", "join", "keys", "lower", "lstrip", "pop", "remove", "replace", "reverse", "rstrip", "setdefault", "sort", "split", "splitlines", "strip", "title", "upper", "values", "capitalize", "casefold", "startswith", "endswith", "find", "rfind", "isalpha", "isdigit", "isalnum", "isspace",
}

ALLOWED_NODES = (
    ast.Module, ast.Expr, ast.Assign, ast.AnnAssign, ast.AugAssign,
    ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Set, ast.Dict,
    ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp, ast.Call,
    ast.keyword, ast.Load, ast.Store, ast.Subscript, ast.Slice,
    ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp, ast.comprehension,
    ast.If, ast.For, ast.While, ast.Break, ast.Continue, ast.Pass,
    ast.FunctionDef, ast.Lambda, ast.arguments, ast.arg, ast.Return, ast.Raise,
    ast.Try, ast.ExceptHandler, ast.Assert, ast.JoinedStr, ast.FormattedValue, ast.Yield, ast.YieldFrom,
    ast.Starred, ast.ClassDef, ast.Attribute, ast.Import, ast.ImportFrom, ast.alias,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd, ast.MatMult,
    ast.Invert, ast.Not, ast.UAdd, ast.USub, ast.And, ast.Or,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot,
    ast.In, ast.NotIn,
)


def validate_code(source: str) -> tuple[bool, str]:
    """Allow a useful Python teaching subset and reject powerful runtime access."""
    if len(source) > MAX_CODE_CHARS:
        return False, f"Code is limited to {MAX_CODE_CHARS:,} characters in this sandbox."
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:
        return False, f"SyntaxError: {exc.msg} (line {exc.lineno})"

    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        return False, "That example is too large for the teaching sandbox."

    class_names: set[str] = set()
    function_names: set[str] = set()
    class_methods: set[str] = set()
    imported_names: set[str] = set()
    imported_modules: dict[str, str] = {}
    # Gather declarations first so functions/classes can be called from anywhere.
    for node in nodes:
        if isinstance(node, ast.ClassDef):
            class_names.add(node.name)
            class_methods.update(n.name for n in node.body if isinstance(n, ast.FunctionDef))
        elif isinstance(node, ast.FunctionDef):
            function_names.add(node.name)
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Lambda):
            function_names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Lambda) and isinstance(node.target, ast.Name):
            function_names.add(node.target.id)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                local_name = alias.asname or root
                imported_names.add(local_name)
                imported_modules[local_name] = root
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                if alias.name != "*":
                    imported_names.add(alias.asname or alias.name)

    callable_names = SAFE_BUILTINS | function_names | class_names | imported_names
    module_aliases = {alias for alias, module in imported_modules.items() if module in SAFE_MODULES}
    module_aliases |= {node.module for node in nodes if isinstance(node, ast.ImportFrom) and node.module in SAFE_MODULES}

    for node in nodes:
        if not isinstance(node, ALLOWED_NODES):
            return False, f"{type(node).__name__} is not enabled in the safe Python runner. Imports are limited to math and statistics; file and network access are disabled."
        if isinstance(node, ast.Name) and node.id.startswith("_"):
            return False, "Names beginning with an underscore are disabled in the safe runner."
        if isinstance(node, ast.FunctionDef):
            if node.name.startswith("_") and node.name != "__init__":
                return False, "Only __init__ is supported as a special method in the safe runner."
            if node.decorator_list or node.returns is not None:
                return False, "Decorators and function annotations are not enabled in the safe runner."
        if isinstance(node, ast.ClassDef):
            if node.name.startswith("_") or node.bases or node.keywords or node.decorator_list:
                return False, "Use a simple class without inheritance or decorators in the safe runner."
        if isinstance(node, ast.arg) and node.arg.startswith("_"):
            return False, "Names beginning with an underscore are disabled in the safe runner."
        if isinstance(node, ast.Constant):
            value = node.value
            if isinstance(value, (str, bytes)) and len(value) > 8_000:
                return False, "String and bytes literals are limited to 8,000 characters."
            if isinstance(value, int) and value.bit_length() > 512:
                return False, "Very large integer literals are disabled in the safe runner."
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in callable_names:
                    return False, f"Calling '{node.func.id}' is not enabled in the safe runner."
            elif isinstance(node.func, ast.Attribute):
                # The full attribute check below only permits safe methods / known class members.
                pass
            else:
                return False, "Only direct calls to approved Python functions and methods are enabled."
            if any(keyword.arg not in {"sep", "end"} for keyword in node.keywords):
                user_callable = (isinstance(node.func, ast.Name) and node.func.id in (function_names | class_names)) or (isinstance(node.func, ast.Attribute) and node.func.attr in class_methods)
                if not user_callable:
                    return False, "Only sep= and end= keyword arguments are supported for built-in calls in the safe runner."
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in {"format", "format_map"}:
                return False, "Private attributes and string format traversal are disabled in the safe runner."
            if not isinstance(node.value, (ast.Name, ast.Subscript)):
                return False, "Chained attribute access is disabled in the safe runner."
            parent = next((candidate for candidate in nodes if any(child is node for child in ast.iter_child_nodes(candidate))), None)
            is_call = isinstance(parent, ast.Call) and parent.func is node
            if isinstance(node.value, ast.Name) and node.value.id in module_aliases:
                module = imported_modules.get(node.value.id, node.value.id)
                if isinstance(parent, ast.ImportFrom):
                    pass
                elif node.attr not in SAFE_MODULES.get(module, set()):
                    return False, f"'{node.attr}' is not in the approved {module} API."
            elif not class_methods and node.attr not in SAFE_METHODS:
                return False, f"The attribute '{node.attr}' is not enabled in the safe runner."
            elif not is_call and not class_names:
                return False, f"Only approved method calls are enabled; '{node.attr}' is not available here."
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in SAFE_MODULES:
                    return False, f"Importing '{alias.name}' is disabled. Only math and statistics are available."
        if isinstance(node, ast.ImportFrom):
            if node.level != 0 or node.module not in SAFE_MODULES:
                return False, f"Importing from '{node.module}' is disabled. Only math and statistics are available."
            allowed = SAFE_MODULES[node.module]
            for alias in node.names:
                if alias.name == "*" or alias.name not in allowed:
                    return False, f"'{alias.name}' is not in the approved {node.module} API."

    return True, ""


RUNNER = r'''import ast, builtins, importlib, io, math, statistics, sys
source = sys.stdin.read()
output = io.StringIO()
output_limit = 20000

def safe_print(*values, sep=" ", end="\n"):
    current = output.tell()
    if current >= output_limit:
        return
    text = str(sep).join(str(value) for value in values) + str(end)
    remaining = output_limit - current
    output.write(text[:remaining])
    if len(text) > remaining:
        output.write("\n… output truncated by PyGuide AI …\n")

def safe_range(*args):
    result = range(*args)
    if len(result) > 10000:
        raise ValueError("range is limited to 10,000 values in the safe runner")
    return result

def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level != 0 or name not in ("math", "statistics"):
        raise ImportError("Only the math and statistics modules are available in this sandbox")
    module = importlib.import_module(name)
    for item in fromlist or ():
        if item == "*" or not hasattr(module, item):
            raise ImportError("That import is not available in this sandbox")
    return module

safe = {name: getattr(builtins, name) for name in (
    "abs", "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "isinstance", "iter", "len", "list", "map", "max", "min", "next", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "pow", "divmod", "ord", "chr",
    "Exception", "ArithmeticError", "AssertionError", "IndexError", "KeyError", "NameError", "NotImplementedError", "RuntimeError", "TypeError", "ValueError", "ZeroDivisionError", "__build_class__"
)}
safe["print"] = safe_print
safe["range"] = safe_range
safe["__import__"] = safe_import
namespace = {"__builtins__": safe, "__name__": "__main__"}
failed = False
try:
    tree = ast.parse(source, mode="exec")
    exec(compile(tree, "<pyguide>", "exec"), namespace, namespace)
except BaseException as exc:
    failed = True
    message = f"{type(exc).__name__}: {exc}"
    if output.tell() < output_limit:
        output.write(message[:max(0, output_limit - output.tell())] + "\n")
sys.stdout.write(output.getvalue())
if failed:
    sys.stderr.write("PYGUIDE_RUNTIME_ERROR")
'''


def _limits() -> None:
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (1_000_000, 1_000_000))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        pass


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def log_message(self, format, *args):
        # Keep the preview console quiet; API errors are returned to the UI.
        return

    def do_POST(self):
        if urlparse(self.path).path != "/api/run":
            self.send_error(404, "Not found")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > 16_000:
            self._json(413, {"ok": False, "error": "Request is empty or too large."})
            return
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            code = payload.get("code", "")
            if not isinstance(code, str):
                raise ValueError("Code must be text.")
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self._json(400, {"ok": False, "error": str(exc)})
            return

        ok, reason = validate_code(code)
        if not ok:
            self._json(200, {"ok": False, "output": reason, "durationMs": 0})
            return
        started = time.perf_counter()
        try:
            completed = subprocess.run(
                [sys.executable, "-I", "-S", "-c", RUNNER],
                input=code,
                text=True,
                capture_output=True,
                timeout=2.5,
                cwd=str(ROOT),
                env={"PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"},
                preexec_fn=_limits if os.name == "posix" else None,
            )
            output = completed.stdout[:MAX_OUTPUT_CHARS]
            if completed.stderr and not output:
                output = completed.stderr[:2_000]
            is_error = "PYGUIDE_RUNTIME_ERROR" in completed.stderr or completed.returncode != 0
            if completed.returncode != 0 and not output:
                output = "Execution stopped unexpectedly. Reduce the work in this snippet and try again."
            self._json(200, {"ok": not is_error, "output": output or "(no output)", "durationMs": round((time.perf_counter() - started) * 1000)})
        except subprocess.TimeoutExpired:
            self._json(200, {"ok": False, "output": "Execution timed out. Check for an infinite or very long loop.", "durationMs": 2500})
        except Exception as exc:
            self._json(500, {"ok": False, "error": f"Runner error: {type(exc).__name__}"})

    def _json(self, status: int, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        super().end_headers()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Serve PyGuide AI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"PyGuide AI running at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

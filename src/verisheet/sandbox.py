"""Docker-backed restricted script execution.

Docker is the isolation boundary; AST checks are only defense in depth. The sandbox fails
closed when Docker is missing rather than silently running untrusted scripts on the host.
"""
from __future__ import annotations
import ast
import shutil
import subprocess
import time
from pathlib import Path

ALLOWED_IMPORTS = {"math", "statistics", "json", "csv", "datetime", "decimal", "pandas", "numpy", "matplotlib"}
FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__", "open", "input", "breakpoint"}
FORBIDDEN_NAMES = {"subprocess", "socket", "requests", "urllib", "http", "os", "sys", "ctypes", "importlib", "shutil", "pickle"}


class RestrictedSandbox:
    """Run a trusted generated script in a non-root, no-network Docker container."""
    def __init__(self, timeout_seconds: int = 20, memory_mb: int = 1024, image_name: str = "verisheet-sandbox:latest"):
        self.timeout_seconds = int(timeout_seconds); self.memory_mb = int(memory_mb)
        if self.timeout_seconds < 1 or self.memory_mb < 128: raise ValueError("Invalid sandbox resource limits")
        self.docker = shutil.which("docker")
        self.image_name = image_name

    def validate_script_ast(self, script_text: str) -> None:
        """Reject disallowed imports, file/process/network access and dynamic execution."""
        try: tree = ast.parse(script_text)
        except SyntaxError as exc: raise ValueError("Script contains invalid Python syntax") from exc
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name.split(".")[0] for alias in node.names] if isinstance(node, ast.Import) else [(node.module or "").split(".")[0]]
                if any(name not in ALLOWED_IMPORTS for name in names): raise ValueError("Script imports a module outside the allowlist")
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else node.func.attr if isinstance(node.func, ast.Attribute) else ""
                if name in FORBIDDEN_CALLS: raise ValueError(f"Forbidden call: {name}")
                if isinstance(node.func, ast.Attribute) and node.func.attr.startswith("__"):
                    raise ValueError("Dunder/reflection access is not allowed")
            if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
                raise ValueError(f"Forbidden module/name: {node.id}")
            if isinstance(node, (ast.Global, ast.Nonlocal)):
                raise ValueError("Global/nonlocal declarations are not allowed")

    def run_script(self, script_path: str | Path, input_dir: str | Path, output_dir: str | Path) -> dict:
        """Run a script in Docker with no network, read-only inputs, resource caps and timeout."""
        script = Path(script_path).resolve(); inputs = Path(input_dir).resolve(); outputs = Path(output_dir).resolve()
        if not script.is_file(): raise FileNotFoundError(script)
        if not inputs.is_dir(): raise NotADirectoryError(inputs)
        outputs.mkdir(parents=True, exist_ok=True)
        self.validate_script_ast(script.read_text(encoding="utf-8"))
        if not self.docker: raise RuntimeError("Docker is required for isolated execution; script was not run on the host")
        start = time.monotonic()
        command = [self.docker, "run", "--rm", "--network=none", "--read-only", "--cap-drop=ALL", "--security-opt=no-new-privileges", "--pids-limit=64", f"--memory={self.memory_mb}m", "--cpus=1", "--user=65534:65534", "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "-v", f"{inputs}:/input:ro", "-v", f"{outputs}:/output:rw", "-v", f"{script}:/work/script.py:ro", "-w", "/work", self.image_name, "python", "/work/script.py"]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=self.timeout_seconds, check=False)
            return {"returncode": completed.returncode, "stdout": completed.stdout[-8000:], "stderr": completed.stderr[-8000:], "duration_seconds": time.monotonic() - start, "isolated": True}
        except subprocess.TimeoutExpired as exc:
            return {"returncode": -1, "stdout": (exc.stdout or "")[-8000:] if isinstance(exc.stdout, str) else "", "stderr": "Sandbox execution timed out", "duration_seconds": time.monotonic() - start, "isolated": True, "timed_out": True}

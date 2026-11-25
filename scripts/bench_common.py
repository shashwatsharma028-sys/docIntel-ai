"""Shared setup for Phase 1 benchmarks: offline SDK patching and loading the BASELINE
implementation (backend/*.py at the accepted Phase 0 checkpoint) side by side with the
current one, so both run in one process, on one machine, against identical inputs."""
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
BASELINE_COMMIT = "2ad08355f9dc79976b9928498ea073f626e73ee3"  # accepted Phase 0 checkpoint

os.environ["GEMINI_API_KEY"] = os.environ["GROQ_API_KEY"] = "offline-benchmark-not-real"
for _target in ("google.genai.Client", "groq.Groq"):
    mock.patch(_target).start()

_BASELINE_DIR = Path(tempfile.mkdtemp(prefix="docintel-baseline-src-"))


def load_baseline(module: str):
    """Import backend/<module>.py exactly as committed at BASELINE_COMMIT, as `baseline_<module>`."""
    src = subprocess.run(["git", "-C", str(ROOT), "show", f"{BASELINE_COMMIT}:backend/{module}.py"],
                         check=True, capture_output=True, text=True).stdout
    path = _BASELINE_DIR / f"baseline_{module}.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"baseline_{module}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def baseline_source_sha(module: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", f"{BASELINE_COMMIT}:backend/{module}.py"],
                          check=True, capture_output=True, text=True).stdout.strip()


class quiet:
    """Silence the implementations' print() progress lines during timing."""
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, *exc):
        sys.stdout.close()
        sys.stdout = self._stdout

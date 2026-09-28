"""Build the native library using only Python's standard library."""
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import argparse

ROOT = Path(__file__).resolve().parent

def library_name():
    return {"darwin": "liboperators.dylib", "win32": "operators.dll"}.get(sys.platform, "liboperators.so")


def compiler_command():
    value = os.environ.get("CXX", "g++" if sys.platform == "win32" else "c++")
    if sys.platform == "win32":
        parts = shlex.split(value, posix=False)
        return [part[1:-1] if len(part)>1 and part[0]==part[-1] and part[0] in "\"'" else part for part in parts]
    return shlex.split(value)


def build(native=False):
    if sys.platform not in ("darwin", "linux", "win32"):
        raise RuntimeError("This build script supports macOS, Linux, and Windows with MinGW-w64 GCC.")
    output = ROOT / library_name()
    compiler = compiler_command()
    if not compiler:
        raise ValueError('CXX must name a compiler')
    command = compiler + [
        "-std=c++17", "-O3", "-Wall", "-Wextra", "-Wpedantic",
        "-dynamiclib" if sys.platform == "darwin" else "-shared",
        "operators.cpp", "-o", str(output)]
    command[len(compiler):len(compiler)] = ["-static"] if sys.platform == "win32" else ["-fPIC"]
    if native:
        command[len(compiler):len(compiler)] = ["-mcpu=native" if sys.platform=='darwin' and __import__('platform').machine()=='arm64' else "-march=native"]
    # Replace only after successful linking, preserving a known-good library on failure.
    with tempfile.TemporaryDirectory(prefix=".native-build-", dir=ROOT) as temporary:
        candidate = Path(temporary) / output.name
        command[-1] = str(candidate.relative_to(ROOT))
        subprocess.run(command, check=True, cwd=ROOT)
        os.replace(candidate, output)
    print("Built:", output)
    return output

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('--native',action='store_true',help='Optimize for this CPU; rebuilt library may not run on older CPUs')
    build(parser.parse_args().native)

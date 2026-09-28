"""One entry point: build, both Python suites, optional native sanitizers.

Run with an interpreter that has NumPy/OpenCV for the metal suite.
Failure in any command stops immediately and returns a nonzero exit code.
"""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from operators.build import compiler_command

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sanitizers',action='store_true')
    parser.add_argument('--demos', action='store_true', help='Run demos and decode every generated PNG')
    parser.add_argument('--native', action='store_true', help='Build specifically for this CPU')
    parser.add_argument('--output', type=Path, default=ROOT/'projects/output'/'verified_current')
    args=parser.parse_args()
    if args.sanitizers and sys.platform == 'win32':
        parser.error('This MinGW Windows workflow supports native assertions; run sanitizers on Linux/macOS.')
    for command in [[sys.executable,str(ROOT/'build.py')]+(['--native'] if args.native else []),
                    [sys.executable,'-m','unittest','discover','-s',str(ROOT),'-t',str(ROOT),'-v']]:
        subprocess.run(command,check=True,cwd=ROOT)
    compiler=compiler_command()
    if not compiler:raise ValueError('CXX must name a compiler')
    with tempfile.TemporaryDirectory(prefix='operator-lab-check-') as directory:
        binary=Path(directory)/('native_checks.exe' if sys.platform=='win32' else 'native_checks')
        flags=['-fsanitize=address,undefined','-fno-omit-frame-pointer'] if args.sanitizers else []
        if sys.platform=='win32': flags += ['-static']
        subprocess.run(compiler+['-std=c++17','-O1','-g']+flags+
                       ['operators/tests/native_checks.cpp','-o',str(binary)],check=True,cwd=ROOT)
        environment=os.environ.copy()
        environment['ASAN_OPTIONS']='halt_on_error=1'
        environment['UBSAN_OPTIONS']='halt_on_error=1:print_stacktrace=1'
        subprocess.run([str(binary)],check=True,env=environment)
    if args.demos:
        from projects.tools.smoke_outputs import run_demos
        run_demos(args.output)
    print('PASS: build, operator and project tests, native assertions'+(' and sanitizers' if args.sanitizers else '')+(' and demo outputs' if args.demos else ''))

if __name__=='__main__':main()

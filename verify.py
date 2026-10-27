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
    # Development harness only; algorithm and replay runtime remain pure C++17.
    with tempfile.TemporaryDirectory(prefix='operator-rgbd-check-') as directory:
        def compile_cpp(name, sources):
            binary = Path(directory) / (name + ('.exe' if sys.platform == 'win32' else ''))
            subprocess.run(compiler + ['-std=c++17', '-O1', '-g'] + flags +
                           ['-Ioperators/cpp/include'] + sources + ['-o', str(binary)],
                           check=True, cwd=ROOT)
            return binary
        for name, sources in [
            ('test_rgbd', ['operators/cpp/tests/test_rgbd.cpp', 'operators/cpp/tests/header_link.cpp']),
            ('test_tracking', ['operators/cpp/tests/test_tracking.cpp']),
            ('test_replay_io', ['operators/cpp/tests/test_replay_io.cpp']),
            ('test_navigation', ['operators/cpp/tests/test_navigation.cpp']),
        ]:
            subprocess.run([str(compile_cpp(name, sources))], check=True, env=environment)
        replay = compile_cpp('rgbd_replay', ['operators/cpp/examples/rgbd_replay.cpp'])
        inputs, outputs = Path(directory)/'inputs', Path(directory)/'outputs'
        subprocess.run([str(replay), '--generate-demo', str(inputs)], check=True, env=environment)
        subprocess.run([str(replay), str(inputs/'calibration.txt'), str(inputs/'frames.txt'), str(outputs)],
                       check=True, env=environment)
        import csv
        with (outputs/'results.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != 34 or any(rows[i]['fresh'] != '0' for i in (6, 7, 8)):
            raise RuntimeError('synthetic replay did not withhold observations during occlusion')
        if rows[29]['state'] != 'lost' or rows[30]['state'] != 'confirming' or rows[32]['state'] != 'tracking':
            raise RuntimeError('synthetic replay lost/reselection contract failed')
        for i, row in enumerate(rows):
            image = (inputs/f'{i:06d}.ppm').read_bytes().split(b'\n', 3)[3]
            if row['fresh'] == '1':
                # Independent ground truth: every measured representative must lie
                # on the red target in the generated input, never the blue occluder.
                u = round(40 + 70*(float(row['x_m']) - i*.01)/float(row['z_m']))
                v = round(24 + 70*float(row['y_m'])/float(row['z_m']))
                if not (0 <= u < 80 and 0 <= v < 48):
                    raise RuntimeError('replay representative outside the image')
                pixel = image[(v*80+u)*3:(v*80+u)*3+3]
                if pixel[0] < 200 or pixel[2] > 80:
                    raise RuntimeError('replay associated the distractor or background')
            bmp = (outputs/f'{i:06d}.bmp').read_bytes()
            if bmp[:2] != b'BM' or int.from_bytes(bmp[18:22], 'little') != 160 or int.from_bytes(bmp[22:26], 'little') != 48:
                raise RuntimeError('invalid replay visualization output')
        if not (outputs/'index.html').is_file():
            raise RuntimeError('missing replay viewer')
        print('PASS: 34-frame synthetic replay, target support, occlusion, reselection and output checks')
        navigation = compile_cpp('navigation_replay', ['operators/cpp/examples/navigation_replay.cpp'])
        nav_output = Path(directory)/'navigation'
        subprocess.run([str(navigation), str(nav_output)], check=True, env=environment)
        import math
        import xml.etree.ElementTree as ET
        scenes = {'clear_follow':100, 'crossing_person':110, 'static_detour':150,
                  'blocked_passage':70, 'sensor_dropout':100, 'target_loss':100}
        for name, count in scenes.items():
            folder = nav_output/name
            with (folder/'results.csv').open() as stream:
                control_rows = list(csv.DictReader(stream))
            if len(control_rows) != count:
                raise RuntimeError('navigation frame count mismatch: ' + name)
            summary = dict(line.split('=',1) for line in (folder/'summary.txt').read_text().splitlines())
            if float(summary['minimum_physical_clearance_m']) <= 0:
                raise RuntimeError('synthetic collision: ' + name)
            for i, row in enumerate(control_rows):
                command = [float(row[k]) for k in ('cmd_x_mps','cmd_y_mps','cmd_yaw_rps')]
                if not all(math.isfinite(v) for v in command) or math.hypot(*command[:2]) > .30000001 or abs(command[1]) > .25000001 or abs(command[2]) > .80000001:
                    raise RuntimeError('invalid command limits: ' + name)
                if row['allowed'] == '0' and any(abs(v) > 1e-10 for v in command):
                    raise RuntimeError('nonzero inhibited command: ' + name)
                if name in ('sensor_dropout','target_loss') and 2 <= float(row['time_s']) < 3 and any(abs(v) > 1e-10 for v in command):
                    raise RuntimeError('sensor/target loss failed to stop: ' + name)
                if name == 'blocked_passage' and row['allowed'] != '0':
                    raise RuntimeError('blocked passage incorrectly released motion')
                ET.parse(folder/f'{i:04d}.svg')
                tokens = (folder/f'{i:04d}.scan').read_text().split()
                if len(tokens) != 729 or tokens[0] != 'OPERATOR_LASER_V1' or tokens[1] != '721' or abs(float(tokens[6])-float(row['time_s'])) > 1e-8:
                    raise RuntimeError('incomplete raw scan record: ' + name)
            if not (folder/'parameters.txt').is_file() or not (folder/'index.html').is_file():
                raise RuntimeError('incomplete navigation diagnostics: ' + name)
        print('PASS: 6 closed-loop synthetic navigation scenarios, 630 raw scans/frames and output checks')
    if args.demos:
        from projects.tools.smoke_outputs import run_demos
        run_demos(args.output)
    print('PASS: build, operator and project tests, native assertions and C++ RGB-D checks'+(' and sanitizers' if args.sanitizers else '')+(' and demo outputs' if args.demos else ''))

if __name__=='__main__':main()

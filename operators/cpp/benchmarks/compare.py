#!/usr/bin/env python3
"""Serial A/B performance audit; standard library only, no camera/ROS needed."""
import argparse
import csv
import hashlib
import json
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path


def command(args, **kwargs):
    return subprocess.run(args, check=True, text=True, capture_output=True, **kwargs).stdout


def fingerprint(root):
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.hpp')):
        digest.update(path.relative_to(root).as_posix().encode() + b'\0' + path.read_bytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', default='253782d', help='Git revision before optimization')
    parser.add_argument('--repo', type=Path)
    parser.add_argument('--include', type=Path, help='Override current include directory')
    parser.add_argument('--output', type=Path, required=True, help='New or empty output directory')
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--iterations', type=int, default=101)
    parser.add_argument('--cxx', default=os.environ.get('CXX', 'c++'))
    args = parser.parse_args()
    if not 1 <= args.rounds <= 20 or not 5 <= args.iterations <= 10000:
        parser.error('rounds must be 1..20; iterations must be 5..10000')
    here = Path(__file__).resolve().parent
    repo = (args.repo or here.parents[2]).resolve()
    current = (args.include or repo/'operators/cpp/include').resolve()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output directory must be new or empty')
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    revision = command(['git', 'rev-parse', '--verify', args.baseline+'^{commit}'], cwd=repo).strip()
    compiler = shlex.split(args.cxx)
    if not compiler or not shutil.which(compiler[0]):
        parser.error('C++ compiler not found')
    flags = ['-std=c++17', '-O3', '-DNDEBUG', '-Wall', '-Wextra', '-Wpedantic', '-Werror']
    manifest = {
        'baseline_revision': revision,
        'current_revision': command(['git', 'rev-parse', 'HEAD'], cwd=repo).strip(),
        'current_tree_dirty': bool(command(['git', 'status', '--porcelain'], cwd=repo)),
        'current_include_sha256': fingerprint(current),
        'benchmark_sha256': hashlib.sha256((here/'benchmark.cpp').read_bytes()).hexdigest(),
        'compiler': command(compiler+['--version']).splitlines()[0],
        'flags': flags, 'platform': platform.platform(), 'architecture': platform.machine(),
        'rounds': args.rounds, 'iterations_per_case_per_round': args.iterations,
        'warmup_calls': 5, 'clock': 'std::chrono::steady_clock',
        'allocation_scope': 'replaceable C++ new calls and cumulative requested bytes; not peak live memory or RSS',
        'limitations': 'Synthetic CPU-only data; no detector inference, camera I/O, ROS, robot, NX power-mode or real-time deadline validation.',
    }
    if platform.system() == 'Darwin':
        for key in ['machdep.cpu.brand_string', 'hw.memsize', 'hw.logicalcpu']:
            try:
                manifest[key] = command(['sysctl', '-n', key]).strip()
            except subprocess.CalledProcessError:
                manifest[key] = 'unavailable'
    elif platform.system() == 'Linux':
        manifest['cpuinfo'] = [line for line in Path('/proc/cpuinfo').read_text().splitlines()
                               if line.startswith(('model name', 'Hardware', 'CPU architecture'))][:8]
        # Record configuration when readable; never alter Jetson power/clocks.
        for name, cmd in [('nvpmodel', ['nvpmodel', '-q']), ('jetson_clocks', ['jetson_clocks', '--show'])]:
            if shutil.which(cmd[0]):
                try:
                    manifest[name] = command(cmd)
                except subprocess.CalledProcessError:
                    manifest[name] = 'query unavailable without additional permission'
    records = {'before': [], 'after': []}
    with tempfile.TemporaryDirectory(prefix='operator-perf-') as temp:
        temp = Path(temp)
        baseline = temp/'baseline'
        paths = command(['git', 'ls-tree', '-rz', '--name-only', revision, '--', 'operators/cpp/include'], cwd=repo).split('\0')
        for name in filter(None, paths):
            relative = Path(name).relative_to('operators/cpp/include')
            if '..' in relative.parts or not name.endswith('.hpp'):
                continue
            target = baseline/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(subprocess.run(['git', 'show', revision+':'+name], check=True, cwd=repo, capture_output=True).stdout)
        if not list(baseline.rglob('*.hpp')):
            raise RuntimeError('baseline has no operator headers')
        manifest['baseline_include_sha256'] = fingerprint(baseline)
        for variant, includes in [('before', baseline), ('after', current)]:
            for program in ['benchmark', 'differential']:
                print('Compile', variant, program, flush=True)
                command(compiler+flags+['-I'+str(includes), str(here/(program+'.cpp')), '-o', str(temp/(program+'-'+variant))])
        before = command([str(temp/'differential-before')])
        after = command([str(temp/'differential-after')])
        if before != after:
            (output/'differential-before.txt').write_text(before)
            (output/'differential-after.txt').write_text(after)
            raise RuntimeError('baseline/current differential results differ; inspect output')
        (output/'differential.txt').write_text(before)
        manifest['differential_exact_match'] = True
        print('Exact differential comparison passed', flush=True)
        for round_index in range(args.rounds):
            # Serial alternating order reduces one-sided thermal/order bias.
            order = ['before', 'after'] if round_index % 2 == 0 else ['after', 'before']
            for variant in order:
                print('Run', round_index+1, '/', args.rounds, variant, flush=True)
                raw = command([str(temp/('benchmark-'+variant)), '--iterations', str(args.iterations)])
                (output/f'round_{round_index+1}_{variant}.csv').write_text(raw)
                rows = list(csv.DictReader(raw.splitlines()))
                if not rows:
                    raise RuntimeError('empty benchmark result')
                records[variant].append({row['case']: row for row in rows})
        names = list(records['before'][0])
        summary = []
        for name in names:
            rows = {variant: [record[name] for record in records[variant]] for variant in records}
            if any(row['digest'] != rows['before'][0]['digest'] for group in rows.values() for row in group):
                raise RuntimeError('semantic benchmark digest differs: '+name)
            row = {'case': name}
            for variant, group in rows.items():
                row[variant+'_p50_ms'] = statistics.median(float(r['p50_ms']) for r in group)
                for column in ['p95_ms', 'p99_ms', 'cold_ms']:
                    row[variant+'_'+column] = max(float(r[column]) for r in group)
                for column in ['new_calls_per_op', 'new_bytes_per_op', 'cold_new_bytes']:
                    row[variant+'_'+column] = statistics.median(float(r[column]) for r in group)
            row['p50_speedup'] = row['before_p50_ms']/row['after_p50_ms']
            summary.append(row)
        with (output/'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summary[0]), lineterminator='\n')
            writer.writeheader(); writer.writerows(summary)
    manifest['all_benchmark_digests_match'] = True
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')
    print('Saved', output, flush=True)
    for row in summary:
        print(f"{row['case']}: {row['before_p50_ms']:.6f} -> {row['after_p50_ms']:.6f} ms ({row['p50_speedup']:.2f}x)")


if __name__ == '__main__':
    main()

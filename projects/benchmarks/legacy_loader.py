"""Allow archived pre-package snapshots to run without rewriting historical files."""
import sys


def execute_legacy(spec, module):
    from operators import vision_ops, robust_geometry, fast_stats
    from projects.metal import inspect_surface
    aliases = {'vision_ops': vision_ops, 'robust_geometry': robust_geometry,
               'fast_stats': fast_stats, 'inspect_surface': inspect_surface}
    previous = {name: sys.modules.get(name) for name in aliases}
    try:
        sys.modules.update(aliases)
        spec.loader.exec_module(module)
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old

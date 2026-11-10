"""Test-only bridge to the research implementation; never imported by genrule."""
import ast
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def functions(root, path, names, extra=None):
    tree = ast.parse((Path(root) / path).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == set(names)
    scope = {'np': np, **(extra or {})}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), scope)
    return SimpleNamespace(**{name: scope[name] for name in names})

def load(root):
    sys.path.insert(0, str(root))
    mods = {name: importlib.import_module('core.' + old) for name, old in {
        'parity':'parity_lib', 'exact':'exact_k_lib', 'row':'row_k_lib',
        'latin':'latin_square_lib', 'flower':'flower_rule_lib',
        'orb':'orb_grid_lib', 'switch':'switch_rule_lib'}.items()}
    on = functions(root, 'core/onset_lib.py', ['first_sustained_crossing', 'mem_threshold'])
    smooth = functions(root, 'scripts/redecode_flower_samples.py', ['median_smooth', 'robust_onset'])
    import torch
    mem = functions(root, 'scripts/parity_memorization_eval_cli.py', ['compute_membership_counts'], {'th':torch})
    mods.update(onset=on, smooth=smooth, mem=mem)
    return SimpleNamespace(**mods)

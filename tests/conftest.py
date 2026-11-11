import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(scope='session')
def api():
    if os.environ.get('GENRULE_LEGACY_ROOT'):
        from legacy_adapter import load
        return load(os.environ['GENRULE_LEGACY_ROOT'])
    paths = {'parity':'rules.parity','exact':'rules.exact_k','row':'rules.row_k',
             'latin':'rules.latin_square','flower':'datasets.images.flower',
             'orb':'datasets.images.orb','switch':'datasets.language.light_switch',
             'onset':'analysis.timescales','smooth':'analysis.timescales',
             'mem':'evaluation.memorization'}
    mods={k:importlib.import_module('genrule.'+v) for k,v in paths.items()}
    mods['smooth']=SimpleNamespace(median_smooth=mods['smooth'].median_smooth,
        robust_onset=mods['smooth'].legacy_robust_onset)
    return SimpleNamespace(**mods)

@pytest.fixture(scope='session')
def arrays():
    with np.load(ROOT/'tests/fixtures/reference.npz',allow_pickle=False) as f:
        yield f

@pytest.fixture(scope='session')
def reference():
    return json.loads((ROOT/'tests/fixtures/reference.json').read_text())

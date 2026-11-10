"""Exercise the public package without the old repository or optional heavy imports."""
import os
import subprocess
import sys
from pathlib import Path


def test_base_import_without_torch_cv2_or_legacy():
    code='''
import sys
from importlib.abc import MetaPathFinder
class BlockImports(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.split('.')[0] in {'torch','cv2','core','scripts','transformers','tensorboard'}:
            raise ImportError('Forbidden dependency: '+fullname)
sys.meta_path.insert(0,BlockImports())
from genrule.rules import parity,exact_k,row_k,latin_square
from genrule.analysis.timescales import persistence_onset
from genrule.datasets.language.light_switch import SwitchTask,render_prompt
from genrule.evaluation.metrics import summarize_outputs
assert len(parity.sample_group_parity_vec(2,6,2))==2
assert render_prompt(SwitchTask())
assert summarize_outputs([True],[False]).novel_valid_yield==1
'''
    subprocess.run([sys.executable,'-c',code],check=True,cwd='/tmp',env=os.environ.copy())

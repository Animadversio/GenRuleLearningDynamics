import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from genrule.analysis.timescales import persistence_onset, withhold_small_valid_counts
from genrule.evaluation.metrics import summarize_outputs, exact_match_mask
from genrule.io.tables import read_image_clock_data

ROOT=Path(__file__).resolve().parents[1]

@pytest.mark.parametrize('values,onset,status',[
    ([0,.6,.7,.8],10,'observed'),
    ([0,.6,.1,.8,.9],20,'observed'),
    ([0,.2,.1,.2],None,'right_censored'),
    ([.5,.5,.5],0,'observed'),
    ([np.nan,np.nan,np.nan],None,'missing_data'),
    ([0,.8,np.nan],None,'missing_data'),
    ([],None,'missing_data'),
])
def test_persistence_boundaries(values,onset,status):
    result=persistence_onset(np.arange(len(values))*10,values,.5)
    assert result.onset==onset and result.status==status
    assert result.budget==(10*(len(values)-1) if values else None)

def test_min_step_and_small_counts():
    v=withhold_small_valid_counts([.1,.2,.3],[29,30,31])
    np.testing.assert_equal(v,[np.nan,.2,.3])
    assert persistence_onset([0,10,20],[1,1,1],.5,min_step=10).onset==10
    assert persistence_onset([0,10],[1,1],.5,min_step=11).status=='missing_data'

@pytest.mark.parametrize('steps,values', [([2,1],[.2,.3]),([1,1],[.2,.3]),([1],[.2,.3]),([float('nan')],[.2])])
def test_bad_time_axes_fail(steps,values):
    with pytest.raises(ValueError): persistence_onset(steps,values,.5)

def test_all_cached_image_onsets():
    rows,curves=read_image_clock_data(ROOT/'paper/data')
    assert len(rows)==24
    for row in rows:
        steps,values=curves[(row['run'],row['metric'])]
        result=persistence_onset(steps,values,float(row['threshold']))
        expected=float(row['onset'])
        assert result.budget==float(row['budget'])
        if np.isfinite(expected):
            assert result.status=='observed' and result.onset==expected, row
        else:
            assert result.status=='right_censored' and result.onset is None,row

def test_metrics_keep_denominators_explicit():
    train=np.array([[0,0],[1,1],[1,1]])
    samples=np.array([[0,0],[1,0],[1,1],[0,1],[np.nan,0]])
    matches=exact_match_mask(train,samples)
    np.testing.assert_equal(matches,[True,False,True,False,False])
    result=summarize_outputs([True,True,True,False,False],matches)
    assert result.n_total==5 and result.n_valid==3 and result.n_train_match==2
    assert result.validity==3/5 and result.training_match_all==2/5
    assert result.training_match_valid==2/3 and result.novel_valid_yield==1/5
    assert result.novel_valid_yield==pytest.approx(result.validity*(1-result.training_match_valid))

def test_empty_and_invalid_output_metrics():
    result=summarize_outputs([False,False],[False,False])
    assert np.isnan(result.training_match_valid)
    assert result.validity==0 and result.novel_valid_yield==0
    empty=summarize_outputs([],[])
    assert empty.n_total==0 and np.isnan(empty.validity)
    with pytest.raises(ValueError):summarize_outputs([False],[True])
    with pytest.raises(ValueError):summarize_outputs([True],[True,False])

def test_membership_above_64_dimensions():
    train=np.zeros((1,100),dtype=int)
    samples=np.zeros((2,100),dtype=int);samples[1,-1]=1
    np.testing.assert_equal(exact_match_mask(train,samples),[True,False])
    np.testing.assert_equal(exact_match_mask(np.empty((0,100)),samples),[False,False])

def test_frozen_artifact_hashes():
    manifest=json.loads((ROOT/'tests/fixtures/provenance.json').read_text())
    for name,digest in manifest['artifact_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name

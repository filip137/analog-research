import copy
import json
import os
from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest
import torch
from scipy import stats

from core.artifacts import sha256_file
from methods.maintenance.recovery import tensor_hash
from hardware.drift.flowers import FlowersLaw, FlowersState
from methods.initial_programming import targets, select_pairs, time_weights
from methods.programming.config import InitialProgramming
from ebl.schema.lifecycle import parse_lifecycle, stage_selections, StageSelection, provenance, verify_provenance
from test_workflow_cmo import cmo_inputs, small, case_doc
from workflow_helpers import run_stage


def parameters():
    # The kernel and shape of the supplied fit, simplified conductance tables
    # keep the statistical test independent of its interpolation implementation.
    return dict(bin_edges=[20., 60., 100., 140., 180.], mean_form='pow',
                alpha=.2891970522649005, tau_mean=None, tc=2.5015562531662336,
                lam=.08960890995253555, m_1h=[-9., -7., -3., -.8],
                s_1h=[13., 10., 6., 4.], c0_sd=[3., 2., 1.5, 1.],
                read_sd=[2.5, 2., 1.3, 1.],
                nig=[.49044451146772505, -.1234968236494366, .19098943429184947, .6855444399359982])


@pytest.fixture
def law():
    torch.set_num_threads(1)
    return FlowersLaw(parameters(), dtype=torch.float64)


def test_gaussian_temporal_covariance_and_mean_match_reference(law):
    state = FlowersState(law, torch.full((60000,), 40., dtype=torch.float64), 2., (17,))
    ages = np.array([3., 60., 300., 3600.])
    values = np.stack([state.read(float(t+2), gaussian=True)[1].numpy() for t in ages], axis=1)
    p = parameters()
    g = 51.6+(40-9)/(79.2/103.2)
    centers = (np.array(p['bin_edges'][:-1])+np.array(p['bin_edges'][1:]))/2
    m, s, c, r = [np.interp(g, centers, p[k])*(79.2/103.2) for k in ('m_1h','s_1h','c0_sd','read_sd')]
    lo, hi = np.minimum.outer(ages, ages), np.maximum.outer(ages, ages)
    cov = s*s*np.log1p(lo/p['tc'])/np.log1p(3600/p['tc'])*(lo/hi)**p['lam']+c*c+np.eye(4)*r*r
    np.testing.assert_allclose(np.cov(values.T), cov, rtol=.035, atol=.12)
    np.testing.assert_allclose(values.mean(0), 40+m*(ages/3600)**p['alpha'], atol=.13)


def test_nig_transform_matches_reference_table(law):
    z = np.linspace(-7, 7, 15001)
    xg = np.linspace(-40, 40, 8001)
    zg, keep = np.unique(stats.norm.ppf(np.clip(stats.norminvgauss(*parameters()['nig']).cdf(xg), 1e-15, 1-1e-15)), return_index=True)
    np.testing.assert_allclose(law.transform(torch.tensor(z)).numpy(), np.interp(z, zg, xg[keep]), atol=1e-12)


def test_rewrite_is_fresh_and_untouched_histories_and_held_reads_preserved(law):
    anchor = torch.full((2, 10000), 40., dtype=torch.float64)
    state = FlowersState(law, anchor, 2., (31,))
    control = FlowersState(law, anchor, 2., (31,))
    old = state.read(62., gaussian=True)[1].clone()
    control.read(62., gaussian=True)
    assert state.read(62., gaussian=True)[1].data_ptr() == state.read(62., gaussian=True)[1].data_ptr()
    mask = torch.zeros_like(anchor, dtype=torch.bool)
    mask[:, :5000] = True
    state.rewrite(anchor, mask, 64.)
    control.advance(64.)
    new = state.read(124., gaussian=True)[1]
    other = control.read(124., gaussian=True)[1]
    assert torch.equal(new[~mask], other[~mask])
    assert abs(np.corrcoef(old[mask].numpy(), new[mask].numpy())[0, 1]) < .035
    with pytest.raises(ValueError, match='backward'):
        state.read(62.)


def test_pair_selection_counts_scales_ties_and_common_mode():
    q = torch.zeros(6)
    observed = torch.tensor([[49.6, 58.6, 48.6, 49.6, 58.6, 48.6], [48.6, 58.6, 48.6, 48.6, 58.6, 48.6]])
    mask = select_pairs(observed, q, torch.tensor([1., 2.]), (2, 3), 1/3)
    assert mask.tolist() == [True, False, False, True, False, False]
    random = select_pairs(observed, q, torch.ones(2), (2, 3), 1/3, random_seed=21)
    assert int(random.sum()) == 2
    assert torch.equal(random, select_pairs(observed, q, torch.ones(2), (2, 3), 1/3, random_seed=21))
    g = targets(torch.linspace(-1, 1, 11), 'P6')
    torch.testing.assert_close((g[0]-g[1])/39.6, torch.linspace(-1, 1, 11), atol=3e-7, rtol=1e-6)
    assert (g >= 9).all() and (g <= 88.2).all()
    assert time_weights([0., 10., 3600.]).sum() == pytest.approx(1.)


def config(tmp_path):
    model = tmp_path/'model.json'
    model.write_text(json.dumps(parameters()))
    return InitialProgramming('a'*64, str(model), sha256_file(model), 'b'*64)


@pytest.mark.parametrize('key,value', [('policy','P8'),('retry_fraction',0),('release_seconds',10),
                                      ('evaluation_seconds',[0,7000]),('canary',1),('extra',1)])
def test_settings_fail_closed(tmp_path,key,value):
    raw = config(tmp_path).to_dict(); raw[key] = value
    with pytest.raises(ValueError): InitialProgramming.parse(raw)


@pytest.mark.parametrize('policy', [f'P{i}' for i in range(7)])
def test_native_stage_source_identity_coverage_cost_and_replay(cmo_inputs,tmp_path,monkeypatch,policy):
    teacher, digest = cmo_inputs
    lc = small(**{'onchip':None,'defects.cases':[case_doc()], 'data.corpus':{'name':'synthetic','sha256':digest},
                  'data.sequence_length':9,'data.max_examples':4,'hwa.arms':[{'id':'digital','method':'none'}]})
    results = {}
    for selection in stage_selections(lc):
        if selection.kind != 'deploy':
            results[selection.stage_id] = run_stage(lc,selection,results,teacher,tmp_path/'parents')
    directory,result = results['hwa__digital']
    source = directory/next(a['path'] for a in result['artifacts'] if a['kind']=='hwa_source')
    corpus = torch.load(Path(os.environ['EBL_CORPUS_ROOT'])/'synthetic.pt', weights_only=False)
    tokens = corpus['splits']['validation'][:18].reshape(2,9).contiguous()
    cfg = replace(config(tmp_path), source_sha256=sha256_file(source), cohort_sha256=tensor_hash(tokens),
                  policy=policy, assignment_seeds=(271001,), validation_sequences=2,
                  evaluation_seconds=(0.,60.,3600.),canary=True)
    from methods.initial_programming import prebias_table
    monkeypatch.setattr('ebl.runtime.initial_programming.prebias_table',
                        lambda law,settings,anchor: prebias_table(law,settings,anchor,points=129,quadrature=128))
    doc = lc.to_dict(); doc['initial_programming'] = cfg.to_dict()
    custom = parse_lifecycle(doc)
    verify_provenance(provenance(lc,'hwa_source',hwa_arm='digital'), custom, 'hwa_source', hwa_arm='digital')
    selection = StageSelection('initial_programming','digital',271001)
    directory,result = run_stage(custom,selection,results,teacher,tmp_path/'run')
    metrics = result['metrics']
    assert metrics['source_unchanged'] and metrics['replay_exact'] and metrics['snapshots'] == 3
    assert metrics['cost']['extra_programs'] > 0 if policy in ('P3','P4','P5') else metrics['cost']['extra_programs'] == 0
    rows = json.loads((directory/'initial_programming_measurements.json').read_text())
    assert [r['seconds'] for r in rows] == [0.,60.,3600.]
    assert all(r['canary'] and r['metrics']['scored_tokens'] == 16 for r in rows)
    saved = torch.load(directory/'checkpoints/initial_programming_final.pt', weights_only=False)
    assert saved['clock_seconds'] == rows[-1]['clock_seconds']

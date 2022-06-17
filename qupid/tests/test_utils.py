import biom
import numpy as np
import pytest

from qupid.casematch import CaseMatchOneToOne, CaseMatchCollection
from qupid import utils


def test_filter_table():
    cm = CaseMatchOneToOne({
        "S0A": {"S1B"},
        "S1A": {"S2B"},
        "S2A": {"S4B"}
    })

    cases = [f"S{i}A" for i in range(3)]
    ctrls = [F"S{i}B" for i in range(10)]
    n = len(cases) + len(ctrls)
    d = 500
    rng = np.random.default_rng()

    data = rng.poisson(5, (d, n))
    table = biom.Table(
        data,
        observation_ids=[f"F{i}" for i in range(d)],
        sample_ids=cases + ctrls
    )
    table_filt, case_ctrls = utils.filter_table(table, cm)

    exp_samps = cm.cases.union(cm.controls)
    assert set(table_filt.ids()) == exp_samps
    assert set(case_ctrls.index) == exp_samps

    for case in cm.cases:
        assert case_ctrls[case] == "case"

    for ctrl in cm.controls:
        assert case_ctrls[ctrl] == "control"


def test_filter_table_by_collection():
    rng = np.random.default_rng()
    d = 500
    n = 50

    samp_ids = [f"S{i}" for i in range(n)]
    obs_ids = [f"F{i}" for i in range(d)]
    data = rng.poisson(5, (d, n))
    table = biom.Table(
        data,
        observation_ids=obs_ids,
        sample_ids=samp_ids
    )

    collection = []
    num_iter = 10
    num_cases = 15
    for i in range(num_iter):
        rand_samps = rng.choice(samp_ids, (num_cases, 2), replace=False)
        rand_samps = {x[0]: {x[1]} for x in rand_samps}
        rand_cm = CaseMatchOneToOne(rand_samps)
        collection.append(rand_cm)

    cm_collection = CaseMatchCollection(collection)
    gen = utils.filter_table_by_collection(table, cm_collection)

    for cm, (table_filt, case_ctrls) in zip(cm_collection, gen):
        exp_samps = cm.cases.union(cm.controls)
        assert set(table_filt.ids()) == exp_samps
        assert set(case_ctrls.index) == exp_samps

        for case in cm.cases:
            assert case_ctrls[case] == "case"

        for ctrl in cm.controls:
            assert case_ctrls[ctrl] == "control"


def test_bad_input():
    with pytest.raises(ValueError) as exc_info:
        utils.filter_table_by_collection("", "")

    exp_err_msg = "table must be of type <class 'biom.table.Table'>!"
    print(exc_info.value) == exp_err_msg

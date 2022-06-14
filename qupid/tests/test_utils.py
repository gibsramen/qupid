import biom
import numpy as np

from qupid import CaseMatchOneToOne
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

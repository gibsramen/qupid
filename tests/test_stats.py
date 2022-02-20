import os

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix

from qupid import CaseMatchOneToMany
import qupid.exceptions as exc
import qupid.stats as stats


@pytest.fixture
def case_match_one_to_one_mock():
    json_in = os.path.join(os.path.dirname(__file__), "data/test.json")
    cm = CaseMatchOneToMany.load_mapping(json_in)
    greedy_cm = cm.greedy_match()

    num_samples = len(greedy_cm.cases) + len(greedy_cm.controls)
    rng = np.random.default_rng(42)

    # Create 2D embedding and use distances for matrix
    case_points = rng.multivariate_normal(
        (2, 1),
        [[1, 1], [1, 1]],
        size=int(num_samples / 2)
    )
    ctrl_points = rng.multivariate_normal(
        (5, 2),
        [[2, 1], [1, 2]],
        size=int(num_samples / 2)
    )
    all_points = np.concatenate([case_points, ctrl_points])
    dm = DistanceMatrix(squareform(pdist(all_points)))

    # https://stackoverflow.com/a/20055748
    # https://stackoverflow.com/a/54277518
    dm.ids = list(greedy_cm.cases) + list(greedy_cm.controls)
    greedy_cm.distance_matrix = dm

    return greedy_cm


def test_permanova(case_match_one_to_one_mock):
    pnova_res = stats.case_control_permanova(case_match_one_to_one_mock, 500)

    exp_index = [
        "method name", "test statistic name", "sample size",
        "number of groups", "test statistic", "p-value",
        "number of permutations"
    ]
    assert list(exp_index) == list(pnova_res.index)


def test_permanova_no_dm(case_match_one_to_one_mock):
    case_match_one_to_one_mock.distance_matrix = None
    with pytest.raises(exc.NoDistanceMatrixError) as exc_info:
        stats.case_control_permanova(case_match_one_to_one_mock)
    exp_msg = "CaseMatch object does not have DistanceMatrix!"
    assert str(exc_info.value) == exp_msg

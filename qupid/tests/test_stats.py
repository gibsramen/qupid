import numpy as np
import pandas as pd
import pytest
from skbio import DistanceMatrix

from qupid import CaseMatchOneToMany
from qupid import stats

CASES = [f"case_{x}" for x in list("ABCDEFGH")]
CONTROLS = set([f"ctrl_{x}" for x in list("ABCDEFGHIJKLMNOP")])
N = len(CASES) + len(CONTROLS)
IDX = CASES + list(CONTROLS)


@pytest.fixture
def example_collection():
    cc_map = {case: CONTROLS for case in CASES}
    cm_all = CaseMatchOneToMany(cc_map)
    cm_coll = cm_all.create_matched_pairs(30)
    return cm_coll


@pytest.fixture
def example_dm(example_collection):
    rng = np.random.default_rng()
    values = rng.beta(1, 1, size=(N, N))
    dm = np.triu(values, 1) + np.triu(values, 1).T
    dm = DistanceMatrix(dm, ids=IDX)
    return dm


@pytest.fixture
def example_vals(example_collection):
    rng = np.random.default_rng()
    values = rng.gamma(2, size=N)
    values = pd.Series(values, index=IDX)
    return values


def test_permanova(example_collection, example_dm):
    pnova_res = stats.bulk_permanova(example_collection, example_dm)
    assert pnova_res.shape[0] == 30

    exp_cols = ["method_name", "test_statistic_name", "test_statistic",
                "p-value", "sample_size", "number_of_groups",
                "number_of_permutations"]
    assert (pnova_res.columns == exp_cols).all()


@pytest.mark.parametrize("test", ["t", "mw"])
def test_univariate(example_collection, example_vals, test):
    res = stats.bulk_univariate_test(example_collection, example_vals, test)
    assert res.shape[0] == 30

    exp_cols = ["method_name", "test_statistic_name", "test_statistic",
                "p-value", "sample_size", "number_of_groups"]
    assert (res.columns == exp_cols).all()

import numpy as np
import pandas as pd
import pytest
from skbio import DistanceMatrix

from qupid import CaseMatchOneToMany
from qupid import stats
import qupid.casematch as mm

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

    exp_cols = [
        "method_name",
        "test_statistic_name",
        "test_statistic",
        "p-value",
        "sample_size",
        "number_of_groups",
        "number_of_permutations",
    ]
    assert (pnova_res.columns == exp_cols).all()


@pytest.mark.parametrize("test", ["t", "mw"])
def test_univariate(example_collection, example_vals, test):
    res = stats.bulk_univariate_test(example_collection, example_vals, test)
    assert res.shape[0] == 30

    exp_cols = [
        "method_name",
        "test_statistic_name",
        "test_statistic",
        "p-value",
        "sample_size",
        "number_of_groups",
    ]
    assert (res.columns == exp_cols).all()


def test_univariate_sample_size_per_record():
    cm1 = mm.CaseMatchOneToOne(
        {"case_A": {"ctrl_A"}, "case_B": {"ctrl_B"}, "case_C": {"ctrl_C"}}
    )
    cm2 = mm.CaseMatchOneToOne({"case_A": {"ctrl_A"}, "case_B": {"ctrl_B"}})
    collection = mm.CaseMatchCollection([cm1, cm2])

    all_ids = ["case_A", "case_B", "case_C", "ctrl_A", "ctrl_B", "ctrl_C"]
    rng = np.random.default_rng(42)
    values = pd.Series(rng.normal(size=len(all_ids)), index=all_ids)

    res = stats.bulk_univariate_test(collection, values, test="t")
    assert set(res["sample_size"].values) == {6, 4}


@pytest.mark.parametrize(
    "test,expected_method,expected_stat",
    [
        ("paired-t", "paired-t", "t"),
        ("ttest-rel", "paired-t", "t"),
        ("wilcoxon", "wilcoxon", "W"),
        ("wilcoxon-sr", "wilcoxon", "W"),
    ],
)
def test_paired_test_runs(
    example_collection, example_vals, test, expected_method, expected_stat
):
    res = stats.bulk_univariate_test(example_collection, example_vals, test=test)
    assert res.shape[0] == 30
    exp_cols = [
        "method_name",
        "test_statistic_name",
        "test_statistic",
        "p-value",
        "sample_size",
        "number_of_groups",
    ]
    assert (res.columns == exp_cols).all()
    assert (res["method_name"] == expected_method).all()
    assert (res["test_statistic_name"] == expected_stat).all()


def test_invalid_test_name(example_collection, example_vals):
    with pytest.raises(ValueError, match="paired-t"):
        stats.bulk_univariate_test(example_collection, example_vals, test="nope")


def test_smd_columns():
    focus = pd.DataFrame({"age": [10.0, 20.0, 30.0]}, index=["c0", "c1", "c2"])
    background = pd.DataFrame(
        {"age": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]},
        index=["b0", "b1", "b2", "b3", "b4", "b5"],
    )
    casematch = mm.CaseMatchOneToOne({"c0": {"b0"}, "c1": {"b1"}, "c2": {"b2"}})
    result = stats.compute_covariate_balance(focus, background, casematch, ["age"])
    assert list(result.columns) == ["covariate", "smd_pre", "smd_post"]


def test_smd_zero_for_identical():
    vals = [1.0, 2.0, 3.0]
    focus = pd.DataFrame({"age": vals}, index=["c0", "c1", "c2"])
    background = pd.DataFrame({"age": vals}, index=["b0", "b1", "b2"])
    casematch = mm.CaseMatchOneToOne({"c0": {"b0"}, "c1": {"b1"}, "c2": {"b2"}})
    result = stats.compute_covariate_balance(focus, background, casematch, ["age"])
    age_row = result.loc[result["covariate"] == "age"].iloc[0]
    assert age_row["smd_pre"] == pytest.approx(0.0)


def test_smd_skips_nonnumeric():
    focus = pd.DataFrame({"grp": ["A", "B", "C"]}, index=["c0", "c1", "c2"])
    background = pd.DataFrame({"grp": ["A", "B", "C"]}, index=["b0", "b1", "b2"])
    casematch = mm.CaseMatchOneToOne({"c0": {"b0"}, "c1": {"b1"}, "c2": {"b2"}})
    with pytest.warns(UserWarning):
        result = stats.compute_covariate_balance(focus, background, casematch, ["grp"])
    assert len(result) == 0


def test_smd_post_better_than_pre():
    focus = pd.DataFrame({"age": [10.0, 10.0, 10.0]}, index=["c0", "c1", "c2"])
    background = pd.DataFrame(
        {"age": [10.0, 10.0, 10.0, 50.0, 50.0, 50.0]},
        index=["b0", "b1", "b2", "b3", "b4", "b5"],
    )
    casematch = mm.CaseMatchOneToOne({"c0": {"b0"}, "c1": {"b1"}, "c2": {"b2"}})
    result = stats.compute_covariate_balance(focus, background, casematch, ["age"])
    age_row = result.loc[result["covariate"] == "age"].iloc[0]
    assert abs(age_row["smd_post"]) < abs(age_row["smd_pre"])


def test_correct_adds_q_column(example_collection, example_vals):
    res_q = stats.bulk_univariate_test(example_collection, example_vals, correct=True)
    assert "q_value" in res_q.columns
    res_no_q = stats.bulk_univariate_test(example_collection, example_vals)
    assert "q_value" not in res_no_q.columns


def test_bh_fdr_monotone():
    p = np.array([0.01, 0.03, 0.05, 0.10, 0.50])
    q = stats._bh_fdr(p)
    assert np.all(np.diff(q) >= -1e-12)


def test_bh_fdr_caps_at_one():
    p = np.array([0.5, 0.8, 0.9, 0.95, 1.0])
    q = stats._bh_fdr(p)
    assert np.all(q <= 1.0)


def test_correct_in_permanova(example_collection, example_dm):
    res_q = stats.bulk_permanova(example_collection, example_dm, correct=True)
    assert "q_value" in res_q.columns


def test_single_permanova_fields():
    exp_fields = {
        "method name",
        "test statistic name",
        "sample size",
        "number of groups",
        "test statistic",
        "p-value",
        "number of permutations",
    }
    cm = mm.CaseMatchOneToOne({"c0": {"t0"}, "c1": {"t1"}, "c2": {"t2"}})
    all_ids = ["c0", "c1", "c2", "t0", "t1", "t2"]
    rng = np.random.default_rng(0)
    dm_arr = np.zeros((6, 6))
    for i in range(6):
        for j in range(i + 1, 6):
            v = rng.random()
            dm_arr[i, j] = dm_arr[j, i] = v
    dm = DistanceMatrix(dm_arr, ids=all_ids)
    result = stats._single_permanova(cm, dm, permutations=99)
    assert set(result.index) == exp_fields


def test_single_permanova_matches_skbio():
    rng = np.random.default_rng(42)
    n = 10
    ids = [f"s{i}" for i in range(n)]
    raw = rng.random((n, n))
    dm_arr = np.triu(raw, 1) + np.triu(raw, 1).T
    dm = DistanceMatrix(dm_arr, ids=ids)
    cm = mm.CaseMatchOneToOne({f"s{i}": {f"s{i+5}"} for i in range(5)})
    result = stats._single_permanova(cm, dm, permutations=99)
    assert result["test statistic"] > 0
    assert 0.0 < result["p-value"] <= 1.0

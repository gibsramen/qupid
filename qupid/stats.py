from __future__ import annotations

from typing import Callable
from warnings import warn

import numpy as np
from joblib import Parallel, delayed
import pandas as pd
import scipy.stats as ss
from skbio import DistanceMatrix
from skbio.stats.distance import permanova as _skbio_permanova

from qupid.casematch import CaseMatchCollection, CaseMatchOneToMany, CaseMatchOneToOne

# Registry of supported univariate tests.
# Dict over if/elif so the CLI can enumerate valid aliases dynamically
# (click.Choice uses _TEST_REGISTRY.keys()) and paired tests can be added
# without touching the dispatch logic.
# Each entry: alias -> (scipy function, display method name, statistic name, paired?).
_TEST_REGISTRY: dict[str, tuple[Callable, str, str, bool]] = {
    "t": (ss.ttest_ind, "t-test", "t", False),
    "ttest": (ss.ttest_ind, "t-test", "t", False),
    "t-test": (ss.ttest_ind, "t-test", "t", False),
    "mw": (ss.mannwhitneyu, "mann-whitney", "U", False),
    "mannwhitney": (ss.mannwhitneyu, "mann-whitney", "U", False),
    "mann-whitney": (ss.mannwhitneyu, "mann-whitney", "U", False),
    "paired-t": (ss.ttest_rel, "paired-t", "t", True),
    "ttest-rel": (ss.ttest_rel, "paired-t", "t", True),
    "paired-ttest": (ss.ttest_rel, "paired-t", "t", True),
    "wilcoxon": (ss.wilcoxon, "wilcoxon", "W", True),
    "wilcoxon-sr": (ss.wilcoxon, "wilcoxon", "W", True),
}


def bulk_permanova(
    casematches: CaseMatchCollection,
    distance_matrix: DistanceMatrix,
    permutations: int = 999,
    n_jobs: int = 1,
    parallel_args: dict = None,
    correct: bool = False,
) -> pd.DataFrame:
    """Evaluate PERMANOVA on multiple case-control mappings.

    :param casematches: Mappings of cases to controls
    :type casematches: qupid.CaseMatchCollection

    :param distance_matrix: Distance matrix of cases and controls
    :type distance_matrix: skbio.DistanceMatrix

    :param permutations: Number of PERMANOVA permutations, defaults to 999
    :type permutations: int

    :param n_jobs: Number of jobs to run in parallel, defaults to 1
        (single CPU).  Pass ``-1`` to use all available cores.
    :type n_jobs: int

    :param parallel_args: Dictionary of arguments to be passed into
        joblib.Parallel. See the documentation for this class at
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
    :type parallel_args: dict

    :param correct: If True, append a ``q_value`` column with
        Benjamini-Hochberg FDR-corrected p-values across iterations,
        defaults to False
    :type correct: bool

    :returns: PERMANOVA results for all mappings
    :rtype: pd.DataFrame
    """
    if parallel_args is None:
        parallel_args = dict()

    pnova_results = Parallel(n_jobs=n_jobs, **parallel_args)(
        delayed(_single_permanova)(cm, distance_matrix, permutations)
        for cm in casematches
    )
    pnova_results = pd.DataFrame.from_records(pnova_results)
    pnova_results.columns = [x.replace(" ", "_") for x in pnova_results.columns]
    pnova_results = pnova_results.sort_values(by="test_statistic", ascending=False)
    col_order = [
        "method_name",
        "test_statistic_name",
        "test_statistic",
        "p-value",
        "sample_size",
        "number_of_groups",
        "number_of_permutations",
    ]
    if correct:
        pnova_results["q_value"] = _bh_fdr(pnova_results["p-value"].values)
        col_order.append("q_value")
    return pnova_results[col_order]


def bulk_univariate_test(
    casematches: CaseMatchCollection,
    values: pd.Series,
    test: str = "t",
    n_jobs: int = 1,
    parallel_args: dict = None,
    correct: bool = False,
) -> pd.DataFrame:
    """Evaluate univariate test on multiple case-control mappings.

    Supports both independent and paired tests. Paired tests (``paired-t``,
    ``wilcoxon``) align each case value with its matched control value by
    the pairing recorded in :class:`CaseMatchOneToOne`, which is more
    powerful than treating them as independent groups.

    :param casematches: Mappings of cases to controls
    :type casematches: qupid.CaseMatchCollection

    :param values: Numeric values to be used for statistical test
    :type values: pd.Series

    :param test: Statistical test to use:

        * ``'t'`` / ``'t-test'`` / ``'ttest'`` — independent t-test
        * ``'mw'`` / ``'mann-whitney'`` — Mann-Whitney U
        * ``'paired-t'`` / ``'ttest-rel'`` — paired t-test (uses matched pairing)
        * ``'wilcoxon'`` / ``'wilcoxon-sr'`` — Wilcoxon signed-rank (uses matched pairing)

        Defaults to ``'t'``.
    :type test: str

    :param n_jobs: Number of jobs to run in parallel, defaults to 1
        (single CPU)
    :type n_jobs: int

    :param parallel_args: Dictionary of arguments to be passed into
        joblib.Parallel. See the documentation for this class at
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
    :type parallel_args: dict

    :param correct: If True, append a ``q_value`` column with
        Benjamini-Hochberg FDR-corrected p-values across iterations,
        defaults to False
    :type correct: bool

    :returns: Test results for all mappings
    :rtype: pd.DataFrame
    """
    test_lower = test.lower()
    if test_lower not in _TEST_REGISTRY:
        raise ValueError(f"test must be one of {sorted(_TEST_REGISTRY)}, got {test!r}")
    test_fn, method_str, stat_str, paired = _TEST_REGISTRY[test_lower]

    if parallel_args is None:
        parallel_args = dict()

    results = Parallel(n_jobs=n_jobs, **parallel_args)(
        delayed(_single_univariate_test)(cm, values, test_fn, paired)
        for cm in casematches
    )
    results = pd.DataFrame.from_records(results)
    results["method_name"] = method_str
    results["test_statistic_name"] = stat_str
    results["sample_size"] = [len(cm.cases) * 2 for cm in casematches]
    results["number_of_groups"] = 2
    results = results.sort_values(by="test_statistic", ascending=False)
    col_order = [
        "method_name",
        "test_statistic_name",
        "test_statistic",
        "p-value",
        "sample_size",
        "number_of_groups",
    ]
    if correct:
        results["q_value"] = _bh_fdr(results["p-value"].values)
        col_order.append("q_value")
    return results[col_order]


def compute_covariate_balance(
    focus: pd.DataFrame,
    background: pd.DataFrame,
    casematch: CaseMatchOneToMany | CaseMatchOneToOne,
    categories: list[str],
) -> pd.DataFrame:
    """Compute standardized mean difference (SMD) per covariate before and
    after matching.

    Pre-matching SMD compares all cases against all background controls.
    Post-matching SMD compares the matched subset of cases against their
    matched controls.  Values near zero indicate good balance; the
    conventional threshold for "well-balanced" is |SMD| < 0.1.

    Numeric and boolean columns use Cohen's d (pooled-SD denominator).
    Non-numeric columns are skipped with a warning.

    :param focus: Metadata for all case samples (rows = samples)
    :type focus: pd.DataFrame

    :param background: Metadata for all control samples (rows = samples)
    :type background: pd.DataFrame

    :param casematch: Matching result (cases and their matched controls)
    :type casematch: CaseMatchOneToMany or CaseMatchOneToOne

    :param categories: Column names to include in the balance report
    :type categories: list[str]

    :returns: DataFrame with columns ``covariate``, ``smd_pre``, ``smd_post``
    :rtype: pd.DataFrame
    """
    matched_cases = focus.loc[list(casematch.cases)]
    matched_controls = background.loc[list(casematch.controls)]

    rows = []
    for cat in categories:
        if cat not in focus.columns or cat not in background.columns:
            warn(f"Category {cat!r} not found in both focus and background; skipping.")
            continue
        col_focus = focus[cat]
        if not _is_numeric_or_bool(col_focus):
            warn(
                f"Category {cat!r} is not numeric or boolean; skipping SMD computation."
            )
            continue
        smd_pre = _smd(col_focus.astype(float), background[cat].astype(float))
        smd_post = _smd(
            matched_cases[cat].astype(float),
            matched_controls[cat].astype(float),
        )
        rows.append({"covariate": cat, "smd_pre": smd_pre, "smd_post": smd_post})

    return pd.DataFrame(rows, columns=["covariate", "smd_pre", "smd_post"])


def _is_numeric_or_bool(series: pd.Series) -> bool:
    """Return True if the series dtype is numeric or boolean."""
    return pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series)


def _smd(g1: pd.Series, g2: pd.Series) -> float:
    """Cohen's d with pooled-SD denominator."""
    mean1, mean2 = g1.mean(), g2.mean()
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_sd = np.sqrt((var1 + var2) / 2.0)
    if pooled_sd == 0:
        return 0.0
    return float((mean1 - mean2) / pooled_sd)


def _bh_fdr(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    q = np.minimum(p * n / ranks, 1.0)
    # Enforce monotonicity right-to-left
    q[order] = np.minimum.accumulate(q[order][::-1])[::-1]
    return q


def _single_permanova(
    casematch: CaseMatchOneToOne, distance_matrix: DistanceMatrix, permutations: int
) -> pd.Series:
    """Evaluate PERMANOVA on single case-control mapping.

    :param casematch: Mapping of cases to controls
    :type casematch: qupid.CaseMatchOneToOne

    :param distance_matrix: Distance matrix of cases and controls
    :type distance_matrix: skbio.DistanceMatrix

    :returns: PERMANOVA results
    :rtype: pd.Series
    """
    cases = list(casematch.cases)
    controls = list(casematch.controls)
    sample_ids = cases + controls
    # Extract submatrix via numpy fancy indexing — skbio's Cython .filter()
    # rejects read-only mmap arrays passed by joblib's loky workers.
    id_to_idx = {sid: i for i, sid in enumerate(distance_matrix.ids)}
    idx = [id_to_idx[s] for s in sample_ids]
    dm_arr = distance_matrix.data[np.ix_(idx, idx)].copy()
    sub_dm = DistanceMatrix(dm_arr, ids=sample_ids)
    grouping = ["case"] * len(cases) + ["control"] * len(controls)
    return _skbio_permanova(sub_dm, grouping, permutations=permutations)


def _single_univariate_test(
    casematch: CaseMatchOneToOne,
    values: pd.Series,
    test_fn: Callable,
    paired: bool = False,
) -> pd.Series:
    """Evaluate univariate test on single case-control mapping.

    :param casematch: Mapping of cases to controls
    :type casematch: qupid.CaseMatchOneToOne

    :param values: Numeric values to be used for statistical test
    :type values: pd.Series

    :param test_fn: Function to use for statistical test
    :type test_fn: Callable

    :param paired: If True, align values by matched pairing order
    :type paired: bool

    :returns: Test results
    :rtype: pd.Series
    """
    if paired:
        # Iterate the case_control_map once to preserve case→control pairing order.
        case_ids: list[str] = []
        ctrl_ids: list[str] = []
        for case, ctrl_set in casematch.case_control_map.items():
            case_ids.append(case)
            ctrl_ids.append(next(iter(ctrl_set)))
        case_vals = values.loc[case_ids].values.ravel()
        ctrl_vals = values.loc[ctrl_ids].values.ravel()
    else:
        case_vals = values.loc[list(casematch.cases)].values.ravel()
        ctrl_vals = values.loc[list(casematch.controls)].values.ravel()
    res = test_fn(case_vals, ctrl_vals)
    return pd.Series([res.statistic, res.pvalue], index=["test_statistic", "p-value"])

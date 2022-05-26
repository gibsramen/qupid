from typing import Callable

from joblib import Parallel, delayed
import pandas as pd
import scipy.stats as ss
from skbio import DistanceMatrix
from skbio.stats.distance import permanova

from qupid.casematch import CaseMatchCollection, CaseMatchOneToOne


def bulk_permanova(
    casematches: CaseMatchCollection,
    distance_matrix: DistanceMatrix,
    permutations: int = 999,
    n_jobs: int = 1,
    parallel_args: dict = None
) -> pd.DataFrame:
    """Evaluate PERMANOVA on multiple case-control mappings.

    :param casematches: Mappings of cases to controls
    :type casematches: qupid.CaseMatchCollection

    :param distance_matrix: Distance matrix of cases and controls
    :type distance_matrix: skbio.DistanceMatrix

    :param permutations: Number of PERMANOVA permutations, defaults to 999
    :type permutations: int

    :param n_jobs: Number of jobs to run in parallel, defaults to 1
        (single CPU)
    :type n_jobs: int

    :param parallel_args: Dictionary of arguments to be passed into
        joblib.Parallel. See the documentation for this class at
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
    :type parallel_args: dict

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
    pnova_results.columns = [
        x.replace(" ", "_") for x in pnova_results.columns
    ]
    pnova_results = pnova_results.sort_values(by="test_statistic",
                                              ascending=False)
    col_order = ["method_name", "test_statistic_name", "test_statistic",
                 "p-value", "sample_size", "number_of_groups",
                 "number_of_permutations"]
    return pnova_results[col_order]


def bulk_univariate_test(
    casematches: CaseMatchCollection,
    values: pd.Series,
    test: str = "t",
    n_jobs: int = 1,
    parallel_args: dict = None
):
    """Evaluate univariate test on multiple case-control mappings.

    :param casematches: Mappings of cases to controls
    :type casematches: qupid.CaseMatchCollection

    :param values: Numeric values to be used for statistical test
    :type values: pd.Series

    :param test: Statistical test to use, either 't' for independent t-test or
        'mw' for Mann-Whitney, defaults to 't'
    :type test: str

    :param n_jobs: Number of jobs to run in parallel, defaults to 1
        (single CPU)
    :type n_jobs: int

    :param parallel_args: Dictionary of arguments to be passed into
        joblib.Parallel. See the documentation for this class at
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
    :type parallel_args: dict

    :returns: Test results for all mappings
    :rtype: pd.DataFrame
    """
    if test in ["t", "ttest", "t-test"]:
        test_fn = ss.ttest_ind
        method_str = "t-test"
        stat_str = "t"
    elif test in ["mw", "mannwhitney", "mann-whitney"]:
        test_fn = ss.mannwhitneyu
        method_str = "mann-whitney"
        stat_str = "U"
    else:
        raise ValueError(
            "test must be either 't' (t-test) or 'mw' (Mann-Whitney)"
        )

    if parallel_args is None:
        parallel_args = dict()

    results = Parallel(n_jobs=n_jobs, **parallel_args)(
        delayed(_single_univariate_test)(cm, values, test_fn)
        for cm in casematches
    )
    results = pd.DataFrame.from_records(results)
    results["method_name"] = method_str
    results["test_statistic_name"] = stat_str
    results["sample_size"] = len(casematches[0].cases) * 2
    results["number_of_groups"] = 2
    results = results.sort_values(by="test_statistic", ascending=False)
    col_order = ["method_name", "test_statistic_name", "test_statistic",
                 "p-value", "sample_size", "number_of_groups"]
    return results[col_order]


def _single_permanova(
    casematch: CaseMatchOneToOne,
    distance_matrix: DistanceMatrix,
    permutations: int
) -> pd.Series:
    """Evaluate PERMANOVA on single case-control mapping.

    :param casematch: Mapping of cases to controls
    :type casematch: qupid.CaseMatchOneToOne

    :param distance_matrix: Distance matrix of cases and controls
    :type distance_matrix: skbio.DistanceMatrix

    :returns: PERMANOVA results
    :rtype: pd.Series
    """
    cases = pd.Series("case", index=list(casematch.cases))
    controls = pd.Series("control", index=list(casematch.controls))
    grouping = pd.concat([cases, controls])
    dm_filt = distance_matrix.filter(grouping.index)
    pnova_res = permanova(dm_filt, grouping, permutations=permutations)
    return pnova_res


def _single_univariate_test(
    casematch: CaseMatchOneToOne,
    values: pd.Series,
    test_fn: Callable
) -> pd.Series:
    """Evaluate univariate test on single case-control mapping.

    :param casematch: Mapping of cases to controls
    :type casematch: qupid.CaseMatchOneToOne

    :param values: Numeric values to be used for statistical test
    :type values: pd.Series

    :param test_fn: Function to use for statistical test
    :type distance_matrix: Callable

    :returns: Test results
    :rtype: pd.Series
    """
    case_vals = values.loc[list(casematch.cases)]
    ctrl_vals = values.loc[list(casematch.controls)]
    res = test_fn(case_vals, ctrl_vals)
    res = pd.Series(res, index=["test_statistic", "p-value"])
    return res

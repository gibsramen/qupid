from functools import partial
from typing import List, Dict
from warnings import warn

import pandas as pd

from .casematch import CaseMatchOneToMany
from . import _exceptions as exc
from . import _casematch_utils as util
from ._descriptions import VALID_ON_FAILURE_OPTS


def match_by_single(
    focus: pd.Series,
    background: pd.Series,
    tolerance: float = None,
    on_failure: str = "raise",
) -> CaseMatchOneToMany:
    """Get matched samples for a single category.

    :param focus: Samples to be matched
    :type focus: pd.Series

    :param background: Metadata to match against
    :type background: pd.Series

    :param tolerance: Tolerance for matching continuous metadata
    :type tolerance: float

    :param on_failure: Whether to 'raise' or 'warn' or 'continue' when no
        matches can be found for a focus sample, defaults to 'raise'
    :type on_failure: str

    :returns: Matched control samples
    :rtype: qupid.CaseMatchOneToMany
    """
    if on_failure.lower() not in VALID_ON_FAILURE_OPTS:
        raise ValueError(
            "Invalid argument for 'on_failure', must be one of "
            f"{VALID_ON_FAILURE_OPTS}"
        )

    if set(focus.index) & set(background.index):
        raise exc.IntersectingSamplesError(focus.index, background.index)

    category_type = util._infer_column_type(focus, background)
    if category_type == "discrete":
        if not util._do_category_values_overlap(focus, background):
            raise exc.DisjointCategoryValuesError(focus, background)
        matcher = util._match_discrete

        if tolerance is not None:
            warn(
                "A tolerance was provided for values inferred to be"
                " discrete. Please check the type of your data."
            )
    else:
        # Only want to pass tolerance if continuous category
        if tolerance is None:
            warn("No tolerance was provided, using 1e-08.")
            tolerance = 1e-08

        matcher = partial(util._match_continuous, tolerance=tolerance)

    matches = dict()
    for f_idx, f_val in focus.iteritems():
        hits = matcher(f_val, background.values)
        if hits.any():
            matches[f_idx] = set(background.index[hits])
        else:
            if on_failure == "raise":
                raise exc.NoMatchesError(f_idx)
            elif on_failure == "warn":
                warn(f"No matches found for {f_idx}")
                matches[f_idx] = set()
            else:
                matches[f_idx] = set()

    metadata = pd.concat([focus, background])
    return CaseMatchOneToMany(matches, metadata)


def match_by_multiple(
    focus: pd.DataFrame,
    background: pd.DataFrame,
    categories: List[str],
    tolerance_map: Dict[str, float] = None,
    on_failure: str = "raise"
) -> CaseMatchOneToMany:
    """Get matched samples for multiple categories.

    :param focus: Samples to be matched
    :type focus: pd.DataFrame

    :param background: Metadata to match against
    :type background: pd.DataFrame

    :param categories: Categories to include as matching criteria
    :type categories: List[str]

    :param tolerance_map: Mapping of tolerances for continuous categories.
        Categories not represented default to 1e-08
    :type tolerance_map: Dict[str, float]

    :param on_failure: Whether to 'raise' or 'ignore' sample for which a match
        cannot be found, defaults to 'raise'
    :type on_failure: str

    :returns: Matched control samples
    :rtype: qupid.CaseMatchOneToMany
    """
    if not util._are_categories_subset(categories, focus):
        raise exc.MissingCategoriesError(categories, "focus", focus)

    if not util._are_categories_subset(categories, background):
        raise exc.MissingCategoriesError(categories, "background",
                                         background)

    tolerance_map = tolerance_map or dict()

    # Match everyone at first
    matches = {i: set(background.index) for i in focus.index}

    for cat in categories:
        tol = tolerance_map.get(cat)
        observed = match_by_single(focus[cat], background[cat],
                                   tol, on_failure).case_control_map
        for fidx, fhits in observed.items():
            # Reduce the matches with successive categories
            matches[fidx] = matches[fidx] & fhits
            if not matches[fidx]:
                if on_failure == "raise":
                    raise exc.NoMoreControlsError()

    metadata = pd.concat([focus, background])
    return CaseMatchOneToMany(matches, metadata)


def shuffle(
    focus: pd.DataFrame,
    background: pd.DataFrame,
    categories: List[str],
    iterations: int = 10,
    tolerance_map: Dict[str, float] = None,
    on_failure: str = "raise",
    strict: bool = True,
    n_jobs: int = 1,
    parallel_args: dict = None
) -> pd.DataFrame:
    """Create multiple case-control matches on several matching criteria.

    :param focus: Samples to be matched
    :type focus: pd.DataFrame

    :param background: Metadata to match against
    :type background: pd.DataFrame

    :param categories: Categories to include as matching criteria
    :type categories: List[str]

    :param tolerance_map: Mapping of tolerances for continuous categories.
        Categories not represented are default to 1e-08
    :type tolerance_map: Dict[str, float]

    :param on_failure: Whether to 'raise' or 'warn' or 'continue' when no
        matches can be found for a focus sample, defaults to 'raise'
    :type on_failure: str

    :param iterations: Number of iterations to run, defaults to 10
    :type iterations: int

    :param strict: Whether to perform strict matching. If True, will throw
        an error if a maximum matching is not found. Otherwise will raise a
        warning. Defaults to True.
    :type strict: bool

    :param n_jobs: Number of jobs to run in parallel, defaults to 1
        (single CPU)
    :type n_jobs: int

    :param parallel_args: Dictionary of arguments to be passed into
        joblib.Parallel. See the documentation for this class at
        https://joblib.readthedocs.io/en/latest/generated/joblib.Parallel.html
    :type parallel_args: dict

    :returns: DataFrame where index is cases and each column represents
        a discrete CaseMatchOneToOne instance
    :rtype: pd.DataFrame
    """
    if on_failure not in VALID_ON_FAILURE_OPTS:
        raise ValueError(
            "Invalid argument for 'on_failure', must be one of "
            f"{VALID_ON_FAILURE_OPTS}"
        )

    cm_one_to_many = match_by_multiple(focus, background, categories,
                                       tolerance_map, on_failure)
    res = cm_one_to_many.create_matched_pairs(
        iterations=iterations,
        strict=strict,
        n_jobs=n_jobs,
        parallel_args=parallel_args
    ).to_dataframe()
    return res

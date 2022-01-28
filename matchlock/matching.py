from functools import partial
from typing import Dict, Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd

from .exceptions import (IntersectingSamplesError,
                         DisjointCategoryValuesError,
                         NoMatchesError)


class Match:
    def __init__(
        cases: pd.Series,
        controls: pd.Series
    ):
        ...

def match_by_single(
    focus: pd.Series,
    background: pd.Series,
    category_type: str,
    tolerance: float = 1e-08,
    on_failure: str = "raise"
) -> Match:
    """Get matched samples for a single category.

    :param focus: Samples to be matched
    :type focus: pd.Series

    :param background: Metadata to match against
    :type background: pd.Series

    :param category_type: One of 'continuous' or 'discrete'
    :type category_type: str

    :param tolerance: Tolerance for matching continuous metadata, defaults to
        1e-08
    :type tolerance: float

    :param on_failure: Whether to 'raise' or 'ignore' sample for which a match
        cannot be found, defaults to 'raise'
    :type on_failure: str

    :returns: Matched control samples
    :rtype: matchlock.Match
    """
    if set(focus.index) & set(background.index):
        raise IntersectingSamplesError(focus.index, background.index)

    if category_type == "discrete":
        if not _do_category_values_overlap(focus, background):
            raise DisjointCategoryValuesError(focus, background)
        matcher = _match_discrete
    elif category_type == "continuous":
        # Only want to pass tolerance if continuous category
        matcher = partial(_match_continuous, tolerance=tolerance)
    else:
        raise ValueError(
            "category_type must be 'continuous' or 'discrete'. "
            f"'{category_type}' is not a valid choice."
        )

    matches = {}
    for f_idx, f_val in focus.iteritems():
        hits = matcher(f_val, background.values)
        if hits.any():
            matches[f_idx] = set(background.index[hits])
        else:
            if on_failure == "raise":
                raise NoMatchesError(f_idx)
            else:
                matches[f_idx] - set()

    return matches

def _do_category_values_overlap(
    focus: pd.Series, background: pd.Series
) -> bool:
    """Check to make sure discrete category values overlap.

    :param focus: Samples to be matched
    :type focus: pd.Series

    :param background: Metadata to match against
    :type background: pd.Series

    :returns: True if there are overlaps, False otherwise
    :rtype: bool
    """
    intersection = set(focus.unique()) & set(background.unique())
    return bool(intersection)

def _match_continuous(
    focus_value: float,
    background_values: npt.NDArray[float],
    tolerance: float,
) -> npt.NDArray[bool]:
    """Find matches to a given float value within tolerance.

    :param focus_value: Value to be matched
    :type focus_value: float

    :param background_values: Values in which to search for matches
    :type background_values: np.ndarray

    :param tolerance: Tolerance with which to evaluate matches
    :type tolerance: float

    :returns: Binary array of matches
    :rtype: np.ndarray
    """
    return np.isclose(background_values, focus_value, atol=tolerance)

def _match_discrete(
    focus_value: str,
    background_values: np.ndarray,
) -> npt.NDArray[bool]:
    """Find matches to a given discrete value.

    :param focus_value: Value to be matched
    :type focus_value: str

    :param background_values: Values in which to search for matches
    :type background_values: np.ndarray

    :returns: Binary array of matches
    :rtype: np.ndarray
    """
    return focus_value == background_values

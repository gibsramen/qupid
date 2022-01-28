from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .exceptions import (IntersectingSamplesError,
                         DisjointCategoryValuesError)


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

    return

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

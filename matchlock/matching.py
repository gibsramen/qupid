from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .exceptions import IntersectingSamplesError


class Match:
    def __init__(self):
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

    :returns: Matched control samples
    :rtype: matchlock.Match
    """
    if set(focus.index) & set(background.index):
        raise IntersectingSamplesError(focus.index, background.index)
    return

import json
from functools import partial
from typing import Dict, List, Sequence, TypeVar

import numpy as np
import pandas as pd

from .exceptions import (IntersectingSamplesError,
                         DisjointCategoryValuesError,
                         NoMatchesError)


DiscreteValue = TypeVar("DiscreteValue", str, bool)
ContinuousValue = TypeVar("ContinuousValue", float, int)


class Match:
    def __init__(
        self,
        case_control_map: Dict[str, set],
        tolerance: float = None
    ):
        """Class storing case-control data & metadata.

        :param case_control_map: Dict of cases to sets of controls
        :type case_control_map: dict(str -> set)

        :param tolerance: Tolerance for continuous categories
        :type tolerance: float
        """
        self.case_control_map = case_control_map
        self.tolerance = tolerance

    @property
    def cases(self) -> List[str]:
        """Get names of cases."""
        return list(self.case_control_map.keys())

    def save_mapping(self, path) -> None:
        """Saves case-control mapping to file as JSON.

        :param path: Location to save
        :type path: os.PathLike
        """
        # Can't serialize sets so we convert to lists
        tmp_cc_map = {k: list(v) for k, v in self.case_control_map.items()}
        with open(path, "w") as f:
            json.dump(tmp_cc_map, f)


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

    matches = dict()
    for f_idx, f_val in focus.iteritems():
        hits = matcher(f_val, background.values)
        if hits.any():
            matches[f_idx] = set(background.index[hits])
        else:
            if on_failure == "raise":
                raise NoMatchesError(f_idx)
            else:
                matches[f_idx] = set()

    return Match(matches)


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
    focus_value: ContinuousValue,
    background_values: Sequence[ContinuousValue],
    tolerance: float,
) -> np.ndarray:
    """Find matches to a given float value within tolerance.

    :param focus_value: Value to be matched
    :type focus_value: str, bool

    :param background_values: Values in which to search for matches
    :type background_values: Sequence

    :param tolerance: Tolerance with which to evaluate matches
    :type tolerance: float

    :returns: Binary array of matches
    :rtype: np.ndarray
    """
    return np.isclose(background_values, focus_value, atol=tolerance)


def _match_discrete(
    focus_value: DiscreteValue,
    background_values: Sequence[DiscreteValue],
) -> np.ndarray:
    """Find matches to a given discrete value.

    :param focus_value: Value to be matched
    :type focus_value: str

    :param background_values: Values in which to search for matches
    :type background_values: Sequence

    :returns: Binary array of matches
    :rtype: np.ndarray
    """
    return np.array(focus_value == background_values)

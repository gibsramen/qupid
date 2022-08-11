import json
from typing import Dict, Sequence, TypeVar

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_bool_dtype
from skbio import DistanceMatrix

from qupid import _exceptions as exc

DiscreteValue = TypeVar("DiscreteValue", str, bool)
ContinuousValue = TypeVar("ContinuousValue", float, int)


def _do_category_values_overlap(focus: pd.Series,
                                background: pd.Series) -> bool:
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


def _are_categories_subset(categories: list, target: pd.DataFrame) -> bool:
    """Check to make sure all categories in map are in target DataFrame.

    :param categories: Metadata categories
    :type categories: list

    :param target: DataFrame to interrogate for categories
    :type target: pd.DataFrame

    :returns: True if all categories are present in target, False otherwise
    :rtype: bool
    """
    return set(categories).issubset(target.columns)


def _match_continuous(
    focus_value: ContinuousValue,
    background_values: Sequence[ContinuousValue],
    tolerance: float,
) -> np.ndarray:
    """Find matches to a given float value within tolerance.

    :param focus_value: Value to be matched
    :type focus_value: float, bool

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


def _load(path: str) -> Dict[str, set]:
    """Load mapping file from JSON as dict.

    :param path: Location of filepath
    :type path: str
    """
    with open(path, "r") as f:
        ccm = json.load(f)
    ccm = {k: set(v) for k, v in ccm.items()}
    return ccm


def _check_one_to_one(case_control_map: dict) -> bool:
    """Check if mapping dict is one-to-one (one control per case)."""
    return all([len(ctrls) == 1 for ctrls in case_control_map.values()])


def _validate_distance_matrix(cases: set, controls: set,
                              dm: DistanceMatrix) -> None:
    """Check to see if all cases and controls in DistanceMatrix."""
    cc_samples = cases.union(controls)
    dm_samples = set(dm.ids)
    missing_samples = cc_samples.difference(dm_samples)
    if missing_samples:
        raise exc.MissingSamplesInDistanceMatrixError(missing_samples)


def _infer_column_type(focus: pd.Series, background: pd.Series) -> str:
    """Determine data types."""
    def check_dtype(col: pd.Series):
        if is_string_dtype(col) or is_bool_dtype(col):
            return "discrete"
        elif is_numeric_dtype(col):
            return "continuous"
        else:
            raise ValueError(f"{col} has wrong column type!")

    col_types = {check_dtype(col) for col in [focus, background]}
    if len(col_types) != 1:
        raise ValueError("Focus and background do not have the same dtype")

    return col_types.pop()

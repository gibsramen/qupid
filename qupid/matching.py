from abc import ABC, abstractmethod
from functools import partial, reduce
import json
from typing import Dict, Sequence, Set, TypeVar, Union

import numpy as np
import pandas as pd
from skbio import DistanceMatrix

from .exceptions import (IntersectingSamplesError,
                         DisjointCategoryValuesError,
                         NoMatchesError,
                         MissingCategoriesError,
                         NoMoreControlsError,
                         NotOneToOneError)


DiscreteValue = TypeVar("DiscreteValue", str, bool)
ContinuousValue = TypeVar("ContinuousValue", float, int)


class _BaseCaseMatch(ABC):
    def __init__(self, case_control_map: Dict[str, set],
                 metadata: Union[pd.Series, pd.DataFrame] = None,
                 distance_matrix: DistanceMatrix = None):
        """Base class storing case-control data & metadata.

        :param case_control_map: Dict of cases to sets of controls
        :type case_control_map: dict(str -> set)

        :param metadata: Metadata associated with cases & controls (optional)
        :type metadata: pd.Series or pd.DataFrame

        :param distance_matrix: Beta-diversity distance matrix of cases and
            controls (optional)
        :type distance_matrix: skbio.DistanceMatrix
        """
        self.case_control_map = case_control_map
        self.metadata = metadata
        self.distance_matrix = distance_matrix

    @property
    def cases(self) -> Set[str]:
        """Get names of cases."""
        return set(self.case_control_map.keys())

    @property
    def controls(self) -> Set[str]:
        """Get names of all controls."""
        ccm = self.case_control_map
        return reduce(lambda x, y: x.union(y), ccm.values())

    def save_mapping(self, path: str) -> None:
        """Saves case-control mapping to file as JSON.

        :param path: Location to save
        :type path: os.PathLike
        """
        # Can't serialize sets so we convert to lists
        tmp_cc_map = {k: list(v) for k, v in self.case_control_map.items()}
        with open(path, "w") as f:
            json.dump(tmp_cc_map, f)

    @classmethod
    @abstractmethod
    def load_mapping(cls, path: str):
        """Create CaseMatch object from JSON file."""

    def __getitem__(self, case_name: str) -> set:
        return self.case_control_map[case_name]


class CaseMatchOneToMany(_BaseCaseMatch):
    def __init__(self, case_control_map: Dict[str, set],
                 metadata: Union[pd.Series, pd.DataFrame] = None,
                 distance_matrix: DistanceMatrix = None):
        """Case match object for mapping one case to multiple controls.

        :param case_control_map: Dict of cases to sets of controls
        :type case_control_map: dict(str -> set)

        :param metadata: Metadata associated with cases & controls (optional)
        :type metadata: pd.Series or pd.DataFrame

        :param distance_matrix: Beta-diversity distance matrix of cases and
            controls (optional)
        :type distance_matrix: skbio.DistanceMatrix
        """
        super().__init__(case_control_map, metadata, distance_matrix)

    @classmethod
    def load_mapping(cls, path: str) -> "CaseMatchOneToMany":
        cm = _load_mapping(path)
        return cls(cm)

    # https://www.python.org/dev/peps/pep-0484/#forward-references
    def greedy_match(self, seed: float = None) -> "CaseMatchOneToOne":
        """Pick controls for each case by naive greedy algorithm.

        NOTE: Can probably improve algorithm with "best" match from tolerance
              in the case of continuous. Later on could account for ordinal
              relationships but that's likely a ways off.

        :param seed: Random seed for greedy matching (optional)
        :type seed: float

        :returns: New CaseMatch object with only one control per case
        :rtype: qupid.CaseMatchOneToOne
        """
        rng = np.random.default_rng(seed)

        # Sort from smallest to largest number of matches
        ordered_ccm = sorted(self.case_control_map.items(),
                             key=lambda x: len(x[1]))

        used_controls = set()
        case_controls = []
        greedy_map = dict()
        for i, (case, controls) in enumerate(ordered_ccm):
            if set(controls).issubset(used_controls):
                remaining = [x[0] for x in ordered_ccm[i:]]
                raise NoMoreControlsError(remaining)
            not_used = list(set(controls) - used_controls)
            random_match = rng.choice(not_used)
            case_controls.append(random_match)
            used_controls.add(random_match)

            greedy_map[case] = {random_match}

        return CaseMatchOneToOne(greedy_map, self.metadata,
                                 self.distance_matrix)


class CaseMatchOneToOne(_BaseCaseMatch):
    def __init__(self, case_control_map: Dict[str, set],
                 metadata: Union[pd.Series, pd.DataFrame] = None,
                 distance_matrix: DistanceMatrix = None):
        """Case match object for mapping one case to one control.

        :param case_control_map: Dict of cases to sets of controls
        :type case_control_map: dict(str -> set)

        :param metadata: Metadata associated with cases & controls (optional)
        :type metadata: pd.Series or pd.DataFrame

        :param distance_matrix: Beta-diversity distance matrix of cases and
            controls (optional)
        :type distance_matrix: skbio.DistanceMatrix
        """
        if not _check_one_to_one(case_control_map):
            raise NotOneToOneError(case_control_map)
        super().__init__(case_control_map, metadata, distance_matrix)

    @classmethod
    def load_mapping(cls, path: str) -> "CaseMatchOneToOne":
        cm = _load_mapping(path)
        if not _check_one_to_one(cm):
            raise NotOneToOneError(cm)
        return cls(cm)


def match_by_single(
    focus: pd.Series,
    background: pd.Series,
    category_type: str,
    tolerance: float = 1e-08,
    on_failure: str = "raise"
) -> CaseMatchOneToMany:
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
    :rtype: qupid.CaseMatchOneToMany
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

    metadata = pd.concat([focus, background])
    return CaseMatchOneToMany(matches, metadata)


def match_by_multiple(
    focus: pd.DataFrame,
    background: pd.DataFrame,
    category_type_map: Dict[str, str],
    tolerance_map: Dict[str, float] = None,
    on_failure: str = "raise"
) -> CaseMatchOneToMany:
    """Get matched samples for multiple categories.

    :param focus: Samples to be matched
    :type focus: pd.DataFrame

    :param background: Metadata to match against
    :type background: pd.DataFrame

    :param category_type_map: Mapping of whether each category is continous or
        discrete. Only included categories will be used.
    :type category_type_map: Dict[str, str]

    :param tolerance_map: Mapping of tolerances for continuous categories.
        Categories not represented are default to 1e-08
    :type tolerance_map: Dict[str, float]

    :param on_failure: Whether to 'raise' or 'ignore' sample for which a match
        cannot be found, defaults to 'raise'
    :type on_failure: str

    :returns: Matched control samples
    :rtype: qupid.CaseMatchOneToMany
    """
    if not _are_categories_subset(category_type_map, focus):
        raise MissingCategoriesError(category_type_map, "focus", focus)

    if not _are_categories_subset(category_type_map, background):
        raise MissingCategoriesError(category_type_map, "background",
                                     background)

    if tolerance_map is None:
        tolerance_map = dict()

    # Match everyone at first
    matches = {i: set(background.index) for i in focus.index}

    for cat, cat_type in category_type_map.items():
        tol = tolerance_map.get(cat, 1e-08)
        observed = match_by_single(focus[cat], background[cat], cat_type,
                                   tol, on_failure).case_control_map
        for fidx, fhits in observed.items():
            # Reduce the matches with successive categories
            matches[fidx] = matches[fidx] & fhits
            if not matches[fidx] and on_failure == "raise":
                raise NoMoreControlsError()

    metadata = pd.concat([focus, background])
    return CaseMatchOneToMany(matches, metadata)


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


def _are_categories_subset(category_map: dict, target: pd.DataFrame) -> bool:
    """Check to make sure all categories in map are in target DataFrame.

    :param category_map: Mapping of category names as keys
    :type category_map: dict

    :param target: DataFrame to interrogate for categories
    :type target: pd.DataFrame

    :returns: True if all categories are present in target, False otherwise
    :rtype: bool
    """
    return set(category_map.keys()).issubset(target.columns)


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


def _load_mapping(path: str) -> Dict[str, set]:
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

from abc import ABC, abstractmethod
from functools import partial, reduce
import json
from typing import Dict, Set, Union

import numpy as np
import pandas as pd
from skbio import DistanceMatrix

from qupid import _exceptions as exc
import qupid._casematch_utils as util


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
        cases = set(case_control_map.keys())
        controls = reduce(lambda x, y: x.union(y), case_control_map.values())
        if distance_matrix is not None:
            util._validate_distance_matrix(cases, controls, distance_matrix)
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

    def __eq__(self, other: "_BaseCaseMatch"):
        return self.case_control_map == other.case_control_map


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
        cm = util._load_mapping(path)
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
                raise exc.NoMoreControlsError(remaining)
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
        if not util._check_one_to_one(case_control_map):
            raise exc.NotOneToOneError(case_control_map)
        super().__init__(case_control_map, metadata, distance_matrix)

    @classmethod
    def load_mapping(cls, path: str) -> "CaseMatchOneToOne":
        cm = util._load_mapping(path)
        if not util._check_one_to_one(cm):
            raise exc.NotOneToOneError(cm)
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
        raise exc.IntersectingSamplesError(focus.index, background.index)

    if category_type == "discrete":
        if not util._do_category_values_overlap(focus, background):
            raise exc.DisjointCategoryValuesError(focus, background)
        matcher = util._match_discrete
    elif category_type == "continuous":
        # Only want to pass tolerance if continuous category
        matcher = partial(util._match_continuous, tolerance=tolerance)
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
                raise exc.NoMatchesError(f_idx)
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
    if not util._are_categories_subset(category_type_map, focus):
        raise exc.MissingCategoriesError(category_type_map, "focus", focus)

    if not util._are_categories_subset(category_type_map, background):
        raise exc.MissingCategoriesError(category_type_map, "background",
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
                raise exc.NoMoreControlsError()

    metadata = pd.concat([focus, background])
    return CaseMatchOneToMany(matches, metadata)
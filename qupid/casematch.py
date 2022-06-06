from abc import ABC, abstractmethod
from functools import partial, reduce
import json
from typing import Dict, Set, Union, List
from warnings import warn

from joblib import Parallel, delayed
import networkx as nx
import pandas as pd

from qupid import _exceptions as exc
from qupid.matching import hopcroft_karp_matching
import qupid._casematch_utils as util


class _BaseCaseMatch(ABC):
    __slots__ = "case_control_map", "metadata"

    def __init__(self, case_control_map: Dict[str, set],
                 metadata: Union[pd.Series, pd.DataFrame] = None):
        """Base class storing case-control data & metadata.

        :param case_control_map: Dict of cases to sets of controls
        :type case_control_map: dict(str -> set)

        :param metadata: Metadata associated with cases & controls (optional)
        :type metadata: pd.Series or pd.DataFrame
        """
        if not self._validate_input(case_control_map):
            raise ValueError("Invalid input!")
        self.case_control_map = case_control_map
        self.metadata = metadata

    @property
    def cases(self) -> Set[str]:
        """Get names of cases."""
        return set(self.case_control_map.keys())

    @property
    def controls(self) -> Set[str]:
        """Get names of all controls."""
        ccm = self.case_control_map
        return reduce(lambda x, y: x.union(y), ccm.values())

    @staticmethod
    def _validate_input(case_control_map: dict) -> bool:
        def is_ctrl_set_valid(ctrls):
            return (
                isinstance(ctrls, set) and
                all(map(lambda x: isinstance(x, str), ctrls))
            )

        cases, ctrls = case_control_map.keys(), case_control_map.values()
        cases_valid = map(lambda x: isinstance(x, str), cases)
        ctrls_valid = map(is_ctrl_set_valid, ctrls)
        return all(cases_valid) and all(ctrls_valid)

    def save(self, path: str) -> None:
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
    def load(cls, path: str):
        """Create CaseMatch object from JSON file."""

    def __getitem__(self, case_name: str) -> set:
        return self.case_control_map[case_name]

    def __eq__(self, other: "_BaseCaseMatch") -> bool:
        return self.case_control_map == other.case_control_map


class CaseMatchOneToMany(_BaseCaseMatch):
    def __init__(self, case_control_map: Dict[str, set],
                 metadata: Union[pd.Series, pd.DataFrame] = None):
        """Case match object for mapping one case to multiple controls.

        :param case_control_map: Dict of cases to sets of controls
        :type case_control_map: dict(str -> set)

        :param metadata: Metadata associated with cases & controls (optional)
        :type metadata: pd.Series or pd.DataFrame
        """
        super().__init__(case_control_map, metadata)

    @classmethod
    def load(cls, path: str) -> "CaseMatchOneToMany":
        cm = util._load(path)
        return cls(cm)

    # https://www.python.org/dev/peps/pep-0484/#forward-references
    def create_matched_pairs(
        self,
        iterations: int = 10,
        strict: bool = True,
        n_jobs: int = 1,
        parallel_args: dict = None
    ) -> List["CaseMatchOneToOne"]:
        """Create multiple matched pairs of cases to controls.

        NOTE: Can probably improve algorithm with "best" match from tolerance
              in the case of continuous. Later on could account for ordinal
              relationships but that's likely a ways off.

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

        :returns: Collection of unique CaseMatchOneToOne objects
        :rtype: qupid.CaseMatchCollection
        """
        if parallel_args is None:
            parallel_args = dict()

        all_matches = set()
        G = nx.Graph(self.case_control_map)

        all_matches = Parallel(n_jobs=n_jobs, **parallel_args)(
            delayed(self._get_cm_one_to_one)(G, strict)
            for i in range(iterations)
        )

        return CaseMatchCollection(list(set(all_matches)))

    @staticmethod
    def _create_matched_pairs_single(G: nx.Graph, cases: set = None,
                                     strict: bool = True) -> dict:
        """Create a single matched pair of cases to controls.

        :param G: Bipartite graph on which to perform matching
        :type G: nx.Graph

        :param cases: Cases in graph
        :type cases: set

        :param strict: Whether to perform strict matching. If True, will throw
            an error if a maximum matching is not found. Otherwise will raise a
            warning. Defaults to True.
        :type strict: bool

        :returns: Mapping of case to single control (set)
        :rtype: dict
        """
        M = hopcroft_karp_matching(G, top_nodes=cases)
        M = {k: {v} for k, v in M.items()}
        if len(M) != len(cases):
            missing = set(cases).difference(M.keys())
            if strict:
                raise exc.NoMoreControlsError(missing)
            else:
                warn("Some cases were not matched to a control.", UserWarning)
        return M

    def _get_cm_one_to_one(self, G: nx.Graph,
                           strict: bool) -> "CaseMatchOneToOne":
        """Get a single matching from a graph as CaseMatchOneToOne.

        :param G: Bipartite graph on which to perform matching
        :type G: nx.Graph

        :param strict: Whether to perform strict matching. If True, will throw
            an error if a maximum matching is not found. Otherwise will raise a
            warning.
        :type strict: bool

        :returns: Set of matches from cases to controls
        :rtype: qupid.CaseMatchOneToOne
        """
        M = self._create_matched_pairs_single(G, cases=self.cases,
                                              strict=strict)
        return CaseMatchOneToOne(M, self.metadata)


class CaseMatchOneToOne(_BaseCaseMatch):
    def __init__(self, case_control_map: Dict[str, set],
                 metadata: Union[pd.Series, pd.DataFrame] = None):
        """Case match object for mapping one case to one control.

        :param case_control_map: Dict of cases to sets of controls
        :type case_control_map: dict(str -> set)

        :param metadata: Metadata associated with cases & controls (optional)
        :type metadata: pd.Series or pd.DataFrame
        """
        if not util._check_one_to_one(case_control_map):
            raise exc.NotOneToOneError(case_control_map)
        super().__init__(case_control_map, metadata)

    @classmethod
    def load(cls, path: str) -> "CaseMatchOneToOne":
        cm = util._load(path)
        if not util._check_one_to_one(cm):
            raise exc.NotOneToOneError(cm)
        return cls(cm)

    def to_series(self) -> pd.Series:
        match_tuples = (
            map(
                lambda y: (y[0], list(y[1])[0]),
                self.case_control_map.items()
            )
        )  # (case, control)
        cases, controls = zip(*match_tuples)
        return pd.Series(controls, index=cases)

    def __hash__(self) -> int:
        return hash(frozenset(
            (k, list(v)[0]) for k, v in self.case_control_map.items()
        ))


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


class CaseMatchCollection:
    def __init__(self, case_matches: List[CaseMatchOneToOne] = None):
        def is_valid_cm(x):
            return isinstance(x, CaseMatchOneToOne)

        if not all(map(is_valid_cm, case_matches)):
            raise ValueError("Entries must all be of type CaseMatchOneToOne!")
        self.case_matches = case_matches

    def to_dataframe(self) -> pd.DataFrame:
        match_series = [x.to_series() for x in self.case_matches]
        df = pd.concat(match_series, axis=1)
        df.index.name = "case_id"
        return df

    @classmethod
    def load(cls, path) -> "CaseMatchCollection":
        df = pd.read_table(path, sep="\t", index_col=0)
        casematches = []
        for col in df.columns:
            mapping = {k: {v} for k, v in df[col].to_dict().items()}
            casematches.append(CaseMatchOneToOne(mapping))
        return cls(casematches)

    def save(self, path) -> None:
        df = self.to_dataframe()
        df.to_csv(path, sep="\t", index=True)

    def __next__(self):
        if self._n >= len(self.case_matches):
            raise StopIteration
        cm = self.case_matches[self._n]
        self._n += 1
        return cm

    def __iter__(self):
        self._n = 0
        return self

    def __len__(self):
        return len(self.case_matches)

    def __getitem__(self, index):
        return self.case_matches[index]

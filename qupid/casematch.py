from abc import ABC, abstractmethod
from functools import reduce
import json
from typing import Dict, Set, Union, List, Callable, Iterator
from warnings import warn

from joblib import Parallel, delayed
import networkx as nx
from numpy.random import SeedSequence
import pandas as pd
from pandas.api.types import is_numeric_dtype

from . import _exceptions as exc
from .matching import hopcroft_karp_matching
from . import _casematch_utils as util


class _BaseCaseMatch(ABC):
    def __init__(self, case_control_map: Dict[str, set]):
        """Base class storing case-control data & metadata.

        :param case_control_map: Dict of cases to sets of controls
        :type case_control_map: dict(str -> set)
        """
        if not self._validate_input(case_control_map):
            raise ValueError("Invalid input!")
        self.case_control_map = case_control_map

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
        super().__init__(case_control_map)
        self.metadata = metadata

    @classmethod
    def load(cls, path: str) -> "CaseMatchOneToMany":
        cm = util._load(path)
        return cls(cm)

    # https://www.python.org/dev/peps/pep-0484/#forward-references
    def create_matched_pairs(
        self,
        iterations: int = 10,
        strict: bool = True,
        seed: int = None,
        n_jobs: int = 1,
        parallel_args: dict = None
    ) -> "CaseMatchCollection":
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

        :param seed: Random seed to use for reproducibility. By default does
            not provide a random seed.
        :type seed: int

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

        # Need to account for parallelization with random seed
        # https://numpy.org/doc/stable/reference/random/parallel.html
        ss = SeedSequence(seed)
        child_states = ss.spawn(iterations)

        all_matches = Parallel(n_jobs=n_jobs, **parallel_args)(
            delayed(self._get_cm_one_to_one)(G, strict, child_state)
            for child_state in child_states
        )

        # Need to sort for reproducibility since calling set is random
        # We call set to remove duplicates so that call is necessary
        cm_list = sorted(list(set(all_matches)))
        return CaseMatchCollection(cm_list, self.metadata)

    def _get_cm_one_to_one(
        self,
        G: nx.Graph,
        strict: bool,
        seed: int
    ) -> "CaseMatchOneToOne":
        """Get a single matching from a graph as CaseMatchOneToOne.

        :param G: Bipartite graph on which to perform matching
        :type G: nx.Graph

        :param strict: Whether to perform strict matching. If True, will throw
            an error if a maximum matching is not found. Otherwise will raise a
            warning.
        :type strict: bool

        :param seed: Random seed to use for reproducibility. By default does
            not provide a random seed.
        :type seed: int

        :returns: Set of matches from cases to controls
        :rtype: qupid.CaseMatchOneToOne
        """
        M = hopcroft_karp_matching(G, top_nodes=self.cases, seed=seed)
        M = {k: {v} for k, v in M.items()}
        if len(M) != len(self.cases):
            missing = set(self.cases).difference(M.keys())
            if strict:
                raise exc.NoMoreControlsError(missing)
            else:
                warn("Some cases were not matched to a control.", UserWarning)
        return CaseMatchOneToOne(M)


class CaseMatchOneToOne(_BaseCaseMatch):
    def __init__(self, case_control_map: Dict[str, set]):
        """Case match object for mapping one case to one control.

        :param case_control_map: Dict of cases to sets of controls
        :type case_control_map: dict(str -> set)
        """
        if not util._check_one_to_one(case_control_map):
            raise exc.NotOneToOneError(case_control_map)
        super().__init__(case_control_map)

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

    def evaluate_match_score(self, metadata: pd.DataFrame,
                             categories: List[str]) -> pd.DataFrame:
        """Evaluates how close matches are by numeric categories.

        NOTE: This approach does a lot of redundant computation if applied
              to a full CaseMatchCollection. The same case-ctrl match score
              will be calculated each time. In the future we should maybe
              move this to only calculate each difference once.

        :param metadata: Sample metadata to compare cases and controls
        :type metadata: pd.DataFrame

        :param categories: Numeric category names in metadata
        :type categories: List[str]

        :returns: DataFrame of case-control matches and scores
        :rtype: pd.DataFrame
        """
        cat_types = map(
            lambda x: is_numeric_dtype(x[1]),
            metadata[categories].items()
        )
        if not all(cat_types):
            raise ValueError("Not all categories are numeric!")

        scores = []
        for case_id, ctrl_id in self.case_control_map.items():
            ctrl_id = list(ctrl_id)[0]
            case_md = metadata.loc[case_id, categories]
            ctrl_md = metadata.loc[ctrl_id, categories]
            diff = pd.Series(case_md.values - ctrl_md.values,
                             name=case_id, index=categories)
            diff["ctrl_id"] = ctrl_id
            scores.append(diff)

        score_df = pd.concat(scores, axis=1).T
        score_df = score_df[["ctrl_id"] + categories]
        score_df = score_df.rename(
            columns={cat: f"{cat}_diff" for cat in categories}
        )
        score_df.index.name = "case_id"
        return score_df

    def __hash__(self) -> int:
        return hash(frozenset(
            (k, list(v)[0]) for k, v in self.case_control_map.items()
        ))

    def __lt__(self, other) -> bool:
        """Used for sorting."""
        this_ccm = self.case_control_map
        other_ccm = other.case_control_map
        for (k1, v1), (k2, v2) in zip(this_ccm.items(), other_ccm.items()):
            v1 = list(v1)[0]
            v2 = list(v2)[0]
            if v1 < v2:
                return True
            if v1 > v2:
                return False
        return False  # Instances are equal

    def __gt__(self, other) -> bool:
        """Used for sorting."""
        this_ccm = self.case_control_map
        other_ccm = other.case_control_map
        for (k1, v1), (k2, v2) in zip(this_ccm.items(), other_ccm.items()):
            v1 = list(v1)[0]
            v2 = list(v2)[0]
            if v1 > v2:
                return True
            if v1 < v2:
                return False
        return False  # Instances are equal


class CaseMatchCollection:
    def __init__(self, case_matches: List[CaseMatchOneToOne],
                 metadata: pd.DataFrame = None):
        """Container for multiple matching sets.

        :param case_matches: List of match sets
        :type case_matches: List[CaseMatchOneToOne]

        :param metadata: Metadata associated with cases & controls (optional)
        :type metadata: pd.Series or pd.DataFrame
        """
        def is_valid_cm(x):
            return isinstance(x, CaseMatchOneToOne)

        if not all(map(is_valid_cm, case_matches)):
            raise ValueError("Entries must all be of type CaseMatchOneToOne!")
        self.case_matches = case_matches
        self.metadata = metadata

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame.

        :returns: DataFrame where index is cases and each column represents a
            discrete CaseMatchOneToOne instance
        :rtype: pd.DataFrame
        """
        match_series = [x.to_series() for x in self.case_matches]
        df = pd.concat(match_series, axis=1)
        df.index.name = "case_id"
        return df

    @classmethod
    def from_dataframe(cls, collection: pd.DataFrame) -> "CaseMatchCollection":
        casematches = []
        for col in collection.columns:
            mapping = {k: {v} for k, v in collection[col].to_dict().items()}
            casematches.append(CaseMatchOneToOne(mapping))
        return cls(casematches)

    def assign_metadata(self, metadata: pd.DataFrame) -> None:
        self.metadata = metadata

    def evaluate_match_scores(self, categories: List[str]):
        if self.metadata is None:
            raise ValueError(
                "CaseMatchCollection does not contain sample metadata! Please "
                "use the assign_metadata method to add metadata before "
                "evaluating match scores."
            )

        def single_match_score(cm_one_to_one):
            score_df = cm_one_to_one.evaluate_match_score(
                self.metadata, categories
            )
            return score_df

        all_match_score_dfs = list(self.apply(single_match_score))
        for i, df in enumerate(all_match_score_dfs):
            df["match_num"] = i

        return pd.concat(all_match_score_dfs)

    @classmethod
    def load(cls, path) -> "CaseMatchCollection":
        """Load from TSV."""
        df = pd.read_table(path, sep="\t", index_col=0)
        return cls.from_dataframe(df)

    def apply(self, func: Callable, *args, **kwargs) -> Iterator:
        """Apply a function to each CaseMatchOneToOne in a collection.

        :param func: Function to call on each CaseMatchOneToOne
        :type func: Callable
        """
        return (func(cm, *args, **kwargs) for cm in self.case_matches)

    def save(self, path) -> None:
        """Save as TSV."""
        df = self.to_dataframe()
        df.to_csv(path, sep="\t", index=True)

    def __iter__(self):
        return (cm for cm in self.case_matches)

    def __len__(self):
        return len(self.case_matches)

    def __getitem__(self, index):
        return self.case_matches[index]

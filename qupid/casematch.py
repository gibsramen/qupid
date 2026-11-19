from __future__ import annotations

from abc import ABC, abstractmethod
from functools import total_ordering
import json
from typing import Callable, Iterator
from warnings import warn

from joblib import Parallel, delayed
import networkx as nx
from numpy.random import SeedSequence
import pandas as pd

from . import _exceptions as exc
from .matching import hopcroft_karp_matching
from . import _casematch_utils as util


class _BaseCaseMatch(ABC):
    __slots__ = "case_control_map", "metadata"

    def __init__(
        self,
        case_control_map: dict[str, set],
        metadata: pd.Series | pd.DataFrame = None,
    ):
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
    def cases(self) -> set[str]:
        """Get names of cases."""
        return set(self.case_control_map.keys())

    @property
    def controls(self) -> set[str]:
        """Get names of all controls."""
        ccm = self.case_control_map
        return set().union(*ccm.values())

    @staticmethod
    def _validate_input(case_control_map: dict) -> bool:
        def is_ctrl_set_valid(ctrls):
            return isinstance(ctrls, set) and all(
                map(lambda x: isinstance(x, str), ctrls)
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
    def __init__(
        self,
        case_control_map: dict[str, set],
        metadata: pd.Series | pd.DataFrame = None,
    ):
        """Case match object for mapping one case to multiple controls.

        :param case_control_map: Dict of cases to sets of controls
        :type case_control_map: dict(str -> set)

        :param metadata: Metadata associated with cases & controls (optional)
        :type metadata: pd.Series or pd.DataFrame
        """
        super().__init__(case_control_map, metadata)

    @classmethod
    def load(cls, path: str) -> CaseMatchOneToMany:
        cm = util._load(path)
        return cls(cm)

    def create_matched_pairs(
        self,
        iterations: int = 10,
        strict: bool = True,
        seed: int = None,
        n_jobs: int = 1,
        parallel_args: dict = None,
    ) -> list[CaseMatchOneToOne]:
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
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")

        if parallel_args is None:
            parallel_args = {}

        G = nx.Graph(self.case_control_map)

        # Need to account for parallelization with random seed
        # https://numpy.org/doc/stable/reference/random/parallel.html
        child_states = SeedSequence(seed).spawn(iterations)

        all_matches = Parallel(n_jobs=n_jobs, **parallel_args)(
            delayed(self._get_cm_one_to_one)(G, strict, child_state)
            for child_state in child_states
        )

        cm_list = sorted(set(all_matches))
        return CaseMatchCollection(cm_list)

    def _hk_round(self, G: nx.Graph, seed, strict: bool) -> dict[str, str]:
        """Run one Hopcroft-Karp round and check coverage.

        :returns: case → control mapping for matched cases only
        :rtype: dict[str, str]
        """
        M = hopcroft_karp_matching(G, top_nodes=self.cases, seed=seed)
        if len(M) == len(self.cases):
            return M

        missing = set(self.cases).difference(M.keys())
        if strict:
            raise exc.NoMoreControlsError(missing)
        warn(
            f"Some cases were not matched to a control: {missing}",
            UserWarning,
        )
        return M

    def _get_cm_one_to_one(
        self, G: nx.Graph, strict: bool, seed
    ) -> "CaseMatchOneToOne":
        """Get a single matching from a graph as CaseMatchOneToOne.

        :param G: Bipartite graph on which to perform matching
        :type G: nx.Graph

        :param strict: Whether to perform strict matching. If True, will throw
            an error if a maximum matching is not found. Otherwise will raise a
            warning.
        :type strict: bool

        :param seed: Random seed for reproducibility.
        :type seed: int

        :returns: Set of matches from cases to controls
        :rtype: qupid.CaseMatchOneToOne
        """
        M = self._hk_round(G, seed, strict)
        return CaseMatchOneToOne({k: {v} for k, v in M.items()}, self.metadata)

    def create_matched_groups(
        self,
        n_controls: int,
        iterations: int = 10,
        strict: bool = True,
        seed: int = None,
        n_jobs: int = 1,
        parallel_args: dict = None,
    ) -> list["CaseMatchOneToMany"]:
        """Create multiple k:1 matched groups (one case → N distinct controls).

        Each returned :class:`CaseMatchOneToMany` assigns exactly *n_controls*
        distinct controls to every case within a single iteration.  Controls
        are never double-assigned within the same iteration — each
        Hopcroft-Karp round removes chosen controls from the candidate pool
        before the next round.

        When ``strict=False`` and the pool is exhausted mid-way, a
        ``UserWarning`` is emitted and affected cases receive fewer than
        *n_controls* controls in that iteration (variable-size control sets).

        :param n_controls: Number of distinct controls to assign per case
        :type n_controls: int

        :param iterations: Number of independent matchings to generate,
            defaults to 10
        :type iterations: int

        :param strict: If True, raise when any case cannot receive
            *n_controls* controls.  If False, emit a warning and return the
            partial matching.  Defaults to True.
        :type strict: bool

        :param seed: Random seed for reproducibility, defaults to None
        :type seed: int

        :param n_jobs: Number of parallel jobs, defaults to 1 (single CPU)
        :type n_jobs: int

        :param parallel_args: Extra kwargs forwarded to
            :class:`joblib.Parallel`, defaults to ``{}``
        :type parallel_args: dict

        :returns: List of unique k:1 matchings (deduplicated across iterations)
        :rtype: list[CaseMatchOneToMany]
        """
        if n_controls < 1:
            raise ValueError(f"n_controls must be >= 1, got {n_controls}")
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        if parallel_args is None:
            parallel_args = {}

        child_states = SeedSequence(seed).spawn(iterations)

        results = Parallel(n_jobs=n_jobs, **parallel_args)(
            delayed(self._get_cm_one_to_n)(n_controls, strict, child_state)
            for child_state in child_states
        )

        seen: set[frozenset] = set()
        unique: list[CaseMatchOneToMany] = []
        for cm in results:
            signature = frozenset(
                (case, frozenset(ctrls)) for case, ctrls in cm.case_control_map.items()
            )
            if signature not in seen:
                seen.add(signature)
                unique.append(cm)
        return unique

    def _get_cm_one_to_n(
        self, n_controls: int, strict: bool, seed
    ) -> "CaseMatchOneToMany":
        """Build one k:1 matching by running Hopcroft-Karp ``n_controls`` times.

        Each round removes its assigned controls from the working graph so the
        next round cannot reuse them.
        """
        G = nx.Graph(self.case_control_map)
        merged: dict[str, set] = {case: set() for case in self.cases}
        for round_seed in seed.spawn(n_controls):
            M = self._hk_round(G, round_seed, strict)
            for case, ctrl in M.items():
                merged[case].add(ctrl)
                G.remove_node(ctrl)
        return CaseMatchOneToMany(merged, self.metadata)


@total_ordering
class CaseMatchOneToOne(_BaseCaseMatch):
    def __init__(
        self,
        case_control_map: dict[str, set],
        metadata: pd.Series | pd.DataFrame = None,
    ):
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
    def load(cls, path: str) -> CaseMatchOneToOne:
        cm = util._load(path)
        if not util._check_one_to_one(cm):
            raise exc.NotOneToOneError(cm)
        return cls(cm)

    def to_series(self) -> pd.Series:
        if not self.case_control_map:
            return pd.Series(dtype=object)
        pairs = [
            (case, next(iter(ctrls))) for case, ctrls in self.case_control_map.items()
        ]
        cases, controls = zip(*pairs)
        return pd.Series(controls, index=cases)

    def _pairs(self) -> list[tuple[str, str]]:
        """Return sorted (case, control) pairs for ordering."""
        return sorted(
            (case, next(iter(ctrls))) for case, ctrls in self.case_control_map.items()
        )

    def __hash__(self) -> int:
        return hash(
            frozenset(
                (case, next(iter(ctrls)))
                for case, ctrls in self.case_control_map.items()
            )
        )

    def __lt__(self, other) -> bool:
        """Used for sorting; @total_ordering fills in the remaining comparisons."""
        return self._pairs() < other._pairs()


class CaseMatchCollection:
    def __init__(self, case_matches: list[CaseMatchOneToOne]):
        """Container for multiple matching sets.

        :param case_matches: List of match sets
        :type case_matches: list[CaseMatchOneToOne]
        """

        def is_valid_cm(x):
            return isinstance(x, CaseMatchOneToOne)

        if not all(map(is_valid_cm, case_matches)):
            raise ValueError("Entries must all be of type CaseMatchOneToOne!")
        self.case_matches = case_matches

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame.

        When all matchings share the same set of cases (i.e.,
        ``create_matched_pairs`` was run with ``strict=True``), every cell is
        populated and the result is a complete (cases × iterations) DataFrame.

        When ``strict=False`` is used, some matchings may cover fewer cases
        than others. ``pd.concat`` unions the indices and fills the gaps with
        ``NaN``, and a ``UserWarning`` naming the affected cases is emitted.
        Use ``strict=True`` (the default) to guarantee NaN-free output.

        :returns: DataFrame where index is cases and each column represents a
            discrete CaseMatchOneToOne instance
        :rtype: pd.DataFrame
        """
        match_series = [x.to_series() for x in self.case_matches]
        df = pd.concat(match_series, axis=1)
        df.index.name = "case_id"
        if df.isna().any().any():
            affected = df.index[df.isna().any(axis=1)].tolist()
            warn(
                "to_dataframe() produced NaN values because partial matchings "
                f"(strict=False) have inconsistent case sets. "
                f"Affected cases: {affected}. Use strict=True to avoid NaN.",
                UserWarning,
            )
        return df

    @classmethod
    def from_dataframe(cls, collection: pd.DataFrame) -> CaseMatchCollection:
        casematches = []
        for col in collection.columns:
            mapping = {k: {v} for k, v in collection[col].to_dict().items()}
            casematches.append(CaseMatchOneToOne(mapping))
        return cls(casematches)

    @classmethod
    def load(cls, path) -> CaseMatchCollection:
        """Load from TSV."""
        df = pd.read_table(path, sep="\t", index_col=0)
        return cls.from_dataframe(df)

    def apply(self, func: Callable) -> Iterator:
        """Apply a function to each CaseMatchOneToOne in a collection.

        :param func: Function to call on each CaseMatchOneToOne
        :type func: Callable
        """
        return (func(cm) for cm in self.case_matches)

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

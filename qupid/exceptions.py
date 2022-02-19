from typing import Collection, Dict, Sequence, Union

import pandas as pd


class IntersectingSamplesError(Exception):
    def __init__(self, group_1: Sequence, group_2: Sequence):
        self.intersecting_samples = set(group_1).intersection(set(group_2))
        self.message = (
            "Sample sets are not disjoint. The following samples "
            f"are shared: {self.intersecting_samples}"
        )
        super().__init__(self.message)


class DisjointCategoryValuesError(Exception):
    def __init__(self, group_1: Sequence, group_2: Sequence):
        self.group_1_values = set(group_1)
        self.group_2_values = set(group_2)
        self.message = (
            "No overlap in discrete category values. "
            f"{self.group_1_values} vs. {self.group_2_values}"
        )
        super().__init__(self.message)


class NoMatchesError(Exception):
    def __init__(self, idx: str):
        self.message = f"No valid matches found for sample {idx}."
        super().__init__(self.message)


class MissingCategoriesError(Exception):
    def __init__(
        self,
        category_map: Dict[str, Union[str, float]],
        target_name: str,
        target_df: pd.DataFrame
    ):
        self.missing_categories = (
            set(category_map.keys()).difference(set(target_df.columns))
        )
        self.message = (
            f"The following categories are missing from {target_name}: "
            f"{self.missing_categories}"
        )
        super().__init__(self.message)


class NoMoreControlsError(Exception):
    def __init__(self, remaining: Collection = None):
        self.remaining = remaining
        self.message = (
            "Prematurely exhausted all matching controls."
        )
        if remaining is not None:
            self.message += f" Remaining cases: {self.remaining}"
        super().__init__(self.message)


class NoDistanceMatrixError(Exception):
    def __init__(self):
        self.message = "CaseMatch object does not have DistanceMatrix!"
        super().__init__(self.message)

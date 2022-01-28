from typing import Sequence


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

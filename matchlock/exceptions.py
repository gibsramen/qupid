from typing import Sequence

class IntersectingSamplesError(Exception):
    def __init__(self, group_1: Sequence, group_2: Sequence):
        self.intersecting_samples = set(group_1).intersection(set(group_2))
        self.message = (
            "Sample sets are not disjoint. The following samples "
            f"are shared: {self.intersecting_samples}"
        )
        super().__init__(self.message)


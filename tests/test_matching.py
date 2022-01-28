import pandas as pd
import pytest

from matchlock.exceptions import IntersectingSamplesError
from matchlock.matching import match_by_single

def test_match_by_single_overlap():
    s1 = pd.Series([1, 2, 3, 4])
    s2 = pd.Series([5, 6, 7, 8])
    s1.index = [f"S{x}" for x in range(4)]
    s2.index = [f"S{x+2}" for x in range(4)]

    exp_intersection = {"S2", "S3"}
    with pytest.raises(IntersectingSamplesError) as exc_info:
        match_by_single(s1, s2, "")
    assert exc_info.value.intersecting_samples == exp_intersection

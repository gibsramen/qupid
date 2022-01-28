import pandas as pd
import pytest

from matchlock.exceptions import (IntersectingSamplesError,
                                  DisjointCategoryValuesError)
import matchlock.matching as mm

def test_match_by_single_sample_overlap():
    s1 = pd.Series([1, 2, 3, 4])
    s2 = pd.Series([5, 6, 7, 8])
    s1.index = [f"S{x}" for x in range(4)]
    s2.index = [f"S{x+2}" for x in range(4)]

    exp_intersection = {"S2", "S3"}
    with pytest.raises(IntersectingSamplesError) as exc_info:
        mm.match_by_single(s1, s2, "")
    assert exc_info.value.intersecting_samples == exp_intersection

def test_match_by_single_no_category_overlap():
    s1 = pd.Series(["a", "b", "c", "d"])
    s2 = pd.Series(["e", "f", "g", "h"])
    s1.index = [f"S{x}A" for x in range(4)]
    s2.index = [f"S{x}B" for x in range(4)]

    exp_grp_1 = {"a", "b", "c", "d"}
    exp_grp_2 = {"e", "f", "g", "h"}
    with pytest.raises(DisjointCategoryValuesError) as exc_info:
        mm.match_by_single(s1, s2, "discrete")
    assert exc_info.value.group_1_values == exp_grp_1
    assert exc_info.value.group_2_values == exp_grp_2

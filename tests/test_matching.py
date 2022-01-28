import numpy as np
import pandas as pd
import pytest

import matchlock.exceptions as mexc
import matchlock.matching as mm


class TestErrors:
    def test_match_by_single_sample_overlap(self):
        s1 = pd.Series([1, 2, 3, 4])
        s2 = pd.Series([5, 6, 7, 8])
        s1.index = [f"S{x}" for x in range(4)]
        s2.index = [f"S{x+2}" for x in range(4)]

        exp_intersection = {"S2", "S3"}
        with pytest.raises(mexc.IntersectingSamplesError) as exc_info:
            mm.match_by_single(s1, s2, "")
        assert exc_info.value.intersecting_samples == exp_intersection

    def test_match_by_single_no_category_overlap(self):
        s1 = pd.Series(["a", "b", "c", "d"])
        s2 = pd.Series(["e", "f", "g", "h"])
        s1.index = [f"S{x}A" for x in range(4)]
        s2.index = [f"S{x}B" for x in range(4)]

        exp_grp_1 = {"a", "b", "c", "d"}
        exp_grp_2 = {"e", "f", "g", "h"}
        with pytest.raises(mexc.DisjointCategoryValuesError) as exc_info:
            mm.match_by_single(s1, s2, "discrete")
        assert exc_info.value.group_1_values == exp_grp_1
        assert exc_info.value.group_2_values == exp_grp_2

    def test_invalid_category_type(self):
        s1 = pd.Series(["a", "b", "c", "d"])
        s2 = pd.Series(["a", "b", "c", "d"])
        s1.index = [f"S{x}A" for x in range(4)]
        s2.index = [f"S{x}B" for x in range(4)]

        exp_msg = (
            "category_type must be 'continuous' or 'discrete'. "
            "'ampharos' is not a valid choice."
        )
        with pytest.raises(ValueError) as exc_info:
            mm.match_by_single(s1, s2, "ampharos")
        assert str(exc_info.value) == exp_msg

    def test_no_matches_discrete_raise(self):
        s1 = pd.Series(["a", "b", "c", "d"])
        s2 = pd.Series(["a", "b", "f", "d"])
        s1.index = [f"S{x}A" for x in range(4)]
        s2.index = [f"S{x}B" for x in range(4)]

        exp_msg = "No valid matches found for sample S2A."
        with pytest.raises(mexc.NoMatchesError) as exc_info:
            mm.match_by_single(s1, s2, "discrete", on_failure="raise")
        assert str(exc_info.value) == exp_msg

    def test_no_matches_continuous_raise(self):
        s1 = pd.Series([1.0, 2.0, 3.0, 4.0])
        s2 = pd.Series([1.4, 3.5, 4.5, 0.5])
        s1.index = [f"S{x}A" for x in range(4)]
        s2.index = [f"S{x}B" for x in range(4)]

        exp_msg = "No valid matches found for sample S1A."
        with pytest.raises(mexc.NoMatchesError) as exc_info:
            mm.match_by_single(s1, s2, "continuous", tolerance=0.5,
                               on_failure="raise")
        assert str(exc_info.value) == exp_msg


class TestMatchers:
    def test_match_continuous(self):
        focus_value = 1.0
        background_values = np.array([1.0, 2.0, 0.1, 0.5, 2.1, -0.1])
        tol = 1.0
        exp_hits = np.array([True, True, True, True, False, False])

        hits = mm._match_continuous(focus_value, background_values, tol)
        assert (exp_hits == hits).all()

    def test_match_discrete(self):
        focus_value = "a"
        background_values = np.array(["a", "b", "c", "a", "a"])
        exp_hits = np.array([True, False, False, True, True])

        hits = mm._match_discrete(focus_value, background_values)
        assert (exp_hits == hits).all()

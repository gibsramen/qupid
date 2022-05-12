import json
import os

import numpy as np
import pandas as pd
import pytest
from skbio import DistanceMatrix

import qupid._exceptions as mexc
import qupid.casematch as mm


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

    def test_match_by_multiple_cat_subset_err(self):
        focus_cat_1 = ["A", "B", "C", "B", "C"]
        focus_cat_2 = [1.0, 2.0, 3.0, 2.5, 4.0]
        bg_cat_1 = ["A", "B", "B", "C", "D", "C", "A"]
        bg_cat_2 = [2.0, 1.0, 2.5, 2.5, 3.5, 4.0, 3.0]

        focus_index = [f"S{x}A" for x in range(5)]
        bg_index = [f"S{x}B" for x in range(7)]

        focus = pd.DataFrame({"cat_1": focus_cat_1, "cat_2": focus_cat_2},
                             index=focus_index)
        bg = pd.DataFrame({"cat_1": bg_cat_1, "cat_2": bg_cat_2,
                           "cat_3": bg_cat_1},
                          index=bg_index)

        cat_type_map = {"cat_1": "discrete", "cat_2": "continuous",
                        "cat_3": "discrete"}
        tol_map = {"cat_2": 1.0}

        with pytest.raises(mexc.MissingCategoriesError) as exc_info:
            mm.match_by_multiple(focus, bg, cat_type_map, tol_map)
        assert exc_info.value.missing_categories == {"cat_3"}
        assert "focus" in str(exc_info.value)

        focus = pd.DataFrame({"cat_1": focus_cat_1, "cat_2": focus_cat_2,
                              "cat_3": focus_cat_1},
                             index=focus_index)
        bg = pd.DataFrame({"cat_1": bg_cat_1, "cat_2": bg_cat_2},
                          index=bg_index)

        with pytest.raises(mexc.MissingCategoriesError) as exc_info:
            mm.match_by_multiple(focus, bg, cat_type_map, tol_map)
        assert exc_info.value.missing_categories == {"cat_3"}
        assert "background" in str(exc_info.value)

    def test_no_more_controls(self):
        data = {
            "S0A": {"S0B", "S1B", "S2B"},
            "S1A": {"S1B", "S2B", "S3B"},
            "S2A": {"S4B"},
            "S3A": {"S4B", "S5B"},
            "S4A": {"S5B"}
        }
        match = mm.CaseMatchOneToMany(data)
        with pytest.raises(mexc.NoMoreControlsError) as exc_info:
            match.greedy_match()

        # Should fail on S3A after S4A & S2A
        exp_remaining = {"S0A", "S1A", "S3A"}
        actual_remaining = set(exc_info.value.remaining)
        assert exp_remaining == actual_remaining

    def test_multiple_no_tol_map(self):
        focus_cat_1 = ["A", "B", "C", "B", "C"]
        focus_cat_2 = [1.0, 2.0, 3.0, 2.5, 4.0]
        bg_cat_1 = ["A", "B", "B", "C", "D", "C", "A"]
        bg_cat_2 = [2.0, 1.0, 2.5, 2.5, 3.5, 4.0, 3.0]

        focus_index = [f"S{x}A" for x in range(5)]
        bg_index = [f"S{x}B" for x in range(7)]

        focus = pd.DataFrame({"cat_1": focus_cat_1, "cat_2": focus_cat_2},
                             index=focus_index)
        bg = pd.DataFrame({"cat_1": bg_cat_1, "cat_2": bg_cat_2},
                          index=bg_index)

        cat_type_map = {"cat_1": "discrete", "cat_2": "continuous"}

        with pytest.raises(mexc.NoMoreControlsError) as exc_info:
            mm.match_by_multiple(focus, bg, cat_type_map)

        exp_msg = "Prematurely exhausted all matching controls."
        assert str(exc_info.value) == exp_msg

    def test_not_one_to_one(self, tmp_path):
        outfile = f"{tmp_path}/dummy.json"
        ccm = {"S1A": {"S2B", "S3B"}, "S2A": {"S1B", "S4B"}, "S3A": {"S5B"}}
        tmp_ccm = {k: list(v) for k, v in ccm.items()}
        with open(outfile, "w") as f:
            json.dump(tmp_ccm, f)

        with pytest.raises(mexc.NotOneToOneError) as exc_info1:
            mm.CaseMatchOneToOne.load_mapping(outfile)

        with pytest.raises(mexc.NotOneToOneError) as exc_info2:
            mm.CaseMatchOneToOne(ccm)

        exp_msg = "The following cases are not one-to-one: ['S1A', 'S2A']"
        assert str(exc_info1.value) == str(exc_info2.value) == exp_msg

    def test_missing_samples_dm(self):
        ccm = {"S1A": {"S2B", "S3B"}, "S2A": {"S1B", "S4B"}, "S3A": {"S5B"}}
        dm = DistanceMatrix(np.zeros([5, 5]))
        dm.ids = ("S1A", "S2A", "S3A", "S1B", "S4B")

        err = mexc.MissingSamplesInDistanceMatrixError
        with pytest.raises(err) as exc_info:
            mm.CaseMatchOneToMany(ccm, distance_matrix=dm)

        exp_msg = (
            "The following samples are missing from the DistanceMatrix: "
        )
        assert exp_msg in str(exc_info.value)
        assert exc_info.value.missing_samples == {"S5B", "S2B", "S3B"}


class TestCaseMatch:
    def test_by_single(self):
        s1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
        s2 = pd.Series([3, 5, 7, 9, 4, 6, 6, 2])
        s1.index = [f"S{x}A" for x in range(8)]
        s2.index = [f"S{x}B" for x in range(8)]

        match = mm.match_by_single(s1, s2, "continuous", 1.0)
        exp_match = {
            "S0A": {"S7B"},
            "S1A": {"S0B", "S7B"},
            "S2A": {"S0B", "S4B", "S7B"},
            "S3A": {"S0B", "S1B", "S4B"},
            "S4A": {"S1B", "S4B", "S5B", "S6B"},
            "S5A": {"S1B", "S2B", "S5B", "S6B"},
            "S6A": {"S2B", "S5B", "S6B"},
            "S7A": {"S2B", "S3B"}
        }
        assert match.case_control_map == exp_match

    def test_by_multiple(self):
        focus_cat_1 = ["A", "B", "C", "B", "C"]
        focus_cat_2 = [1.0, 2.0, 3.0, 2.5, 4.0]
        bg_cat_1 = ["A", "B", "B", "C", "D", "C", "A"]
        bg_cat_2 = [2.0, 1.0, 2.5, 2.5, 3.5, 4.0, 3.0]

        focus_index = [f"S{x}A" for x in range(5)]
        bg_index = [f"S{x}B" for x in range(7)]

        focus = pd.DataFrame({"cat_1": focus_cat_1, "cat_2": focus_cat_2},
                             index=focus_index)
        bg = pd.DataFrame({"cat_1": bg_cat_1, "cat_2": bg_cat_2},
                          index=bg_index)

        cat_type_map = {"cat_1": "discrete", "cat_2": "continuous"}
        tol_map = {"cat_2": 1.0}

        match = mm.match_by_multiple(focus, bg, cat_type_map, tol_map)
        exp_match = {
            "S0A": {"S0B"},
            "S1A": {"S1B", "S2B"},
            "S2A": {"S3B", "S5B"},
            "S3A": {"S2B"},
            "S4A": {"S5B"}
        }
        assert match.case_control_map == exp_match

    def test_save_mapping(self, tmp_path):
        outpath = os.path.join(tmp_path, "test.json")

        cc_map = {"S1A": {"S2B", "S4B"}, "S3A": {"S6B", "S2B"}}
        match = mm.CaseMatchOneToMany(cc_map)
        match.save_mapping(outpath)
        with open(outpath, "r") as f:
            content = json.load(f)
            assert set(content["S1A"]) == {"S2B", "S4B"}
            assert set(content["S3A"]) == {"S6B", "S2B"}

    def test_load_mapping(self, tmp_path):
        inpath = os.path.join(tmp_path, "test.json")

        cc_map = {"S1A": ["S2B", "S4B"], "S3A": ["S6B", "S2B"]}
        with open(inpath, "w") as f:
            json.dump(cc_map, f)

        match = mm.CaseMatchOneToMany.load_mapping(inpath)
        assert match["S1A"] == {"S2B", "S4B"}
        assert match["S3A"] == {"S6B", "S2B"}

    def test_case_match_eq(self):
        cc_map1 = {"S1A": {"S2B", "S4B"}, "S3A": {"S6B", "S2B"}}
        cc_map2 = {"S1A": {"S2B", "S4B"}, "S3A": {"S6B", "S2B"}}

        cm1 = mm.CaseMatchOneToMany(cc_map1)
        cm2 = mm.CaseMatchOneToMany(cc_map2)
        assert cm1 == cm2

    def test_load_one_to_one(self, tmp_path):
        outfile = f"{tmp_path}/dummy.json"
        ccm = {"S1A": {"S2B"}, "S2A": {"S1B"}, "S3A": {"S5B"}}
        tmp_ccm = {k: list(v) for k, v in ccm.items()}
        with open(outfile, "w") as f:
            json.dump(tmp_ccm, f)

        cm = mm.CaseMatchOneToOne.load_mapping(outfile)
        assert cm.cases == {"S1A", "S2A", "S3A"}
        assert cm.controls == {"S2B", "S1B", "S5B"}

    def test_properties(self):
        s1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
        s2 = pd.Series([3, 5, 7, 9, 4, 6, 6, 2, 10, 11])
        s1.index = [f"S{x}A" for x in range(8)]
        s2.index = [f"S{x}B" for x in range(10)]

        match = mm.match_by_single(s1, s2, "continuous", 1.0)

        exp_controls = set(s2.index[:-2])
        assert match.controls == exp_controls

        exp_cases = set(s1.index)
        assert match.cases == exp_cases

    def test_greedy_match(self):
        json_in = os.path.join(os.path.dirname(__file__), "data/test.json")
        match = mm.CaseMatchOneToMany.load_mapping(json_in)
        greedy_cm = match.greedy_match()

        assert isinstance(greedy_cm, mm.CaseMatchOneToOne)
        assert len(greedy_cm.cases) == 6
        assert len(greedy_cm.controls) == 6

    def test_on_failure_ignore(self):
        s1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 100])
        s2 = pd.Series([3, 5, 7, 9, 4, 6, 6, 2, 50])
        s1.index = [f"S{x}A" for x in range(9)]
        s2.index = [f"S{x}B" for x in range(9)]

        match = mm.match_by_single(s1, s2, "continuous", 1.0,
                                   on_failure="ignore")
        exp_match = {
            "S0A": {"S7B"},
            "S1A": {"S0B", "S7B"},
            "S2A": {"S0B", "S4B", "S7B"},
            "S3A": {"S0B", "S1B", "S4B"},
            "S4A": {"S1B", "S4B", "S5B", "S6B"},
            "S5A": {"S1B", "S2B", "S5B", "S6B"},
            "S6A": {"S2B", "S5B", "S6B"},
            "S7A": {"S2B", "S3B"},
            "S8A": set()
        }
        assert match.case_control_map == exp_match
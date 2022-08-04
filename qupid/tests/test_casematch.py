import json
import os

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import qupid._exceptions as mexc
import qupid.casematch as mm
from qupid import match_by_single, match_by_multiple


class TestErrors:
    def test_match_by_single_sample_overlap(self):
        s1 = pd.Series([1, 2, 3, 4])
        s2 = pd.Series([5, 6, 7, 8])
        s1.index = [f"S{x}" for x in range(4)]
        s2.index = [f"S{x+2}" for x in range(4)]

        exp_intersection = {"S2", "S3"}
        with pytest.raises(mexc.IntersectingSamplesError) as exc_info:
            match_by_single(s1, s2)
        assert exc_info.value.intersecting_samples == exp_intersection

    def test_match_by_single_no_category_overlap(self):
        s1 = pd.Series(["a", "b", "c", "d"])
        s2 = pd.Series(["e", "f", "g", "h"])
        s1.index = [f"S{x}A" for x in range(4)]
        s2.index = [f"S{x}B" for x in range(4)]

        exp_grp_1 = {"a", "b", "c", "d"}
        exp_grp_2 = {"e", "f", "g", "h"}
        with pytest.raises(mexc.DisjointCategoryValuesError) as exc_info:
            match_by_single(s1, s2)
        assert exc_info.value.group_1_values == exp_grp_1
        assert exc_info.value.group_2_values == exp_grp_2

    def test_no_matches_discrete_raise(self):
        s1 = pd.Series(["a", "b", "c", "d"])
        s2 = pd.Series(["a", "b", "f", "d"])
        s1.index = [f"S{x}A" for x in range(4)]
        s2.index = [f"S{x}B" for x in range(4)]

        exp_msg = "No valid matches found for sample S2A."
        with pytest.raises(mexc.NoMatchesError) as exc_info:
            match_by_single(s1, s2, on_failure="raise")
        assert str(exc_info.value) == exp_msg

    def test_no_matches_continuous_raise(self):
        s1 = pd.Series([1.0, 2.0, 3.0, 4.0])
        s2 = pd.Series([1.4, 3.5, 4.5, 0.5])
        s1.index = [f"S{x}A" for x in range(4)]
        s2.index = [f"S{x}B" for x in range(4)]

        exp_msg = "No valid matches found for sample S1A."
        with pytest.raises(mexc.NoMatchesError) as exc_info:
            match_by_single(s1, s2, tolerance=0.5,
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

        cats = ["cat_1", "cat_2", "cat_3"]
        tol_map = {"cat_2": 1.0}

        with pytest.raises(mexc.MissingCategoriesError) as exc_info:
            match_by_multiple(focus, bg, cats, tol_map)
        assert exc_info.value.missing_categories == {"cat_3"}
        assert "focus" in str(exc_info.value)

        focus = pd.DataFrame({"cat_1": focus_cat_1, "cat_2": focus_cat_2,
                              "cat_3": focus_cat_1},
                             index=focus_index)
        bg = pd.DataFrame({"cat_1": bg_cat_1, "cat_2": bg_cat_2},
                          index=bg_index)

        with pytest.raises(mexc.MissingCategoriesError) as exc_info:
            match_by_multiple(focus, bg, cats, tol_map)
        assert exc_info.value.missing_categories == {"cat_3"}
        assert "background" in str(exc_info.value)

    def test_no_more_controls_strict(self):
        data = {
            "S0A": {"S0B", "S1B", "S2B"},
            "S1A": {"S1B", "S2B", "S3B"},
            "S2A": {"S4B"},
            "S3A": {"S4B", "S5B"},
            "S4A": {"S5B"}
        }
        match = mm.CaseMatchOneToMany(data)
        with pytest.raises(mexc.NoMoreControlsError):
            match.create_matched_pairs()

    def test_no_more_controls_no_strict(self):
        data = {
            "S0A": {"S0B", "S1B", "S2B"},
            "S1A": {"S1B", "S2B", "S3B"},
            "S2A": {"S4B"},
            "S3A": {"S4B", "S5B"},
            "S4A": {"S5B"}
        }
        match = mm.CaseMatchOneToMany(data)
        with pytest.warns(UserWarning) as warn_info:
            match.create_matched_pairs(strict=False)

        exp_msg = "Some cases were not matched to a control."
        assert str(warn_info[0].message) == exp_msg

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

        cats = ["cat_1", "cat_2"]

        with pytest.raises(mexc.NoMoreControlsError) as exc_info:
            match_by_multiple(focus, bg, cats)

        exp_msg = "Prematurely exhausted all matching controls."
        assert str(exc_info.value) == exp_msg

    def test_not_one_to_one(self, tmp_path):
        outfile = f"{tmp_path}/dummy.json"
        ccm = {"S1A": {"S2B", "S3B"}, "S2A": {"S1B", "S4B"}, "S3A": {"S5B"}}
        tmp_ccm = {k: list(v) for k, v in ccm.items()}
        with open(outfile, "w") as f:
            json.dump(tmp_ccm, f)

        with pytest.raises(mexc.NotOneToOneError) as exc_info1:
            mm.CaseMatchOneToOne.load(outfile)

        with pytest.raises(mexc.NotOneToOneError) as exc_info2:
            mm.CaseMatchOneToOne(ccm)

        exp_msg = "The following cases are not one-to-one: ['S1A', 'S2A']"
        assert str(exc_info1.value) == str(exc_info2.value) == exp_msg

    def test_bad_collection_input(self):
        cm = mm.CaseMatchOneToOne({"A": {"X"}, "B": {"Y"}})
        with pytest.raises(ValueError) as exc_info:
            mm.CaseMatchCollection(["A", "B", cm, 5])

        exp_msg = "Entries must all be of type CaseMatchOneToOne!"
        assert str(exc_info.value) == exp_msg

    @pytest.mark.parametrize(
        "test_input", [
            {"A": {"X"}, "B": "Y"},
            {"A": {5}, "B": {"Y"}},
            {5: {"X"}, "B": {"Y"}},
        ]
    )
    def test_invalid_input(self, test_input):
        with pytest.raises(ValueError) as exc_info:
            mm.CaseMatchOneToOne(test_input)

        exp_err_msg = "Invalid input!"
        assert str(exc_info.value) == exp_err_msg


class TestCaseMatch:
    def test_by_single(self):
        s1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
        s2 = pd.Series([3, 5, 7, 9, 4, 6, 6, 2])
        s1.index = [f"S{x}A" for x in range(8)]
        s2.index = [f"S{x}B" for x in range(8)]

        match = match_by_single(s1, s2, 1.0)
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

        cats = ["cat_1", "cat_2"]
        tol_map = {"cat_2": 1.0}

        match = match_by_multiple(focus, bg, cats, tol_map)
        exp_match = {
            "S0A": {"S0B"},
            "S1A": {"S1B", "S2B"},
            "S2A": {"S3B", "S5B"},
            "S3A": {"S2B"},
            "S4A": {"S5B"}
        }
        assert match.case_control_map == exp_match

    def test_lt(self):
        cm_1 = mm.CaseMatchOneToOne({
            "S0A": {"S1B"},
            "S1A": {"S2B"},
            "S2A": {"S3B"},
            "S3A": {"S4B"}
        })
        cm_2 = mm.CaseMatchOneToOne({
            "S0A": {"S1B"},
            "S1A": {"S2B"},
            "S2A": {"S4B"},
            "S3A": {"S3B"}
        })
        assert cm_1 < cm_2

    def test_bool_column_type(self):
        focus_cat_1 = ["A", "B", "C", "B", "C"]
        focus_cat_2 = [True, False, False, True, False]
        bg_cat_1 = ["A", "B", "B", "C", "D", "C", "A"]
        bg_cat_2 = [True, False, True, False, True, False, False]

        focus_index = [f"S{x}A" for x in range(5)]
        bg_index = [f"S{x}B" for x in range(7)]

        focus = pd.DataFrame({"cat_1": focus_cat_1, "cat_2": focus_cat_2},
                             index=focus_index)
        bg = pd.DataFrame({"cat_1": bg_cat_1, "cat_2": bg_cat_2},
                          index=bg_index)

        cats = ["cat_1", "cat_2"]

        match = match_by_multiple(focus, bg, cats)
        exp_match = {
            "S0A": {"S0B"},
            "S1A": {"S1B"},
            "S2A": {"S3B", "S5B"},
            "S3A": {"S2B"},
            "S4A": {"S3B", "S5B"},
        }
        assert match.case_control_map == exp_match

    def test_save(self, tmp_path):
        outpath = os.path.join(tmp_path, "test.json")

        cc_map = {"S1A": {"S2B", "S4B"}, "S3A": {"S6B", "S2B"}}
        match = mm.CaseMatchOneToMany(cc_map)
        match.save(outpath)
        with open(outpath, "r") as f:
            content = json.load(f)
            assert set(content["S1A"]) == {"S2B", "S4B"}
            assert set(content["S3A"]) == {"S6B", "S2B"}

    def test_load(self, tmp_path):
        inpath = os.path.join(tmp_path, "test.json")

        cc_map = {"S1A": ["S2B", "S4B"], "S3A": ["S6B", "S2B"]}
        with open(inpath, "w") as f:
            json.dump(cc_map, f)

        match = mm.CaseMatchOneToMany.load(inpath)
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

        cm = mm.CaseMatchOneToOne.load(outfile)
        assert cm.cases == {"S1A", "S2A", "S3A"}
        assert cm.controls == {"S2B", "S1B", "S5B"}

    def test_properties(self):
        s1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
        s2 = pd.Series([3, 5, 7, 9, 4, 6, 6, 2, 10, 11])
        s1.index = [f"S{x}A" for x in range(8)]
        s2.index = [f"S{x}B" for x in range(10)]

        match = match_by_single(s1, s2, 1.0)

        exp_controls = set(s2.index[:-2])
        assert match.controls == exp_controls

        exp_cases = set(s1.index)
        assert match.cases == exp_cases

    def test_create_matched_pairs(self):
        json_in = os.path.join(os.path.dirname(__file__), "data/test.json")
        match = mm.CaseMatchOneToMany.load(json_in)
        all_matched_pairs = match.create_matched_pairs(iterations=1000)
        assert isinstance(all_matched_pairs, mm.CaseMatchCollection)

        # Should be 36 matches
        match_df = all_matched_pairs.to_dataframe()
        assert match_df.shape == (len(match.cases), 36)

    def test_get_cm_one_to_one(self):
        json_in = os.path.join(os.path.dirname(__file__), "data/test.json")
        match = mm.CaseMatchOneToMany.load(json_in)
        G = nx.Graph(match.case_control_map)

        matched_pairs = match._get_cm_one_to_one(G, False, None)

        assert isinstance(matched_pairs, mm.CaseMatchOneToOne)

        ctrl_lens = [
            len(v) == 1
            for k, v in matched_pairs.case_control_map.items()
        ]
        assert all(ctrl_lens)

    def test_on_failure_continue(self):
        s1 = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 100])
        s2 = pd.Series([3, 5, 7, 9, 4, 6, 6, 2, 50])
        s1.index = [f"S{x}A" for x in range(9)]
        s2.index = [f"S{x}B" for x in range(9)]

        match = match_by_single(s1, s2, 1.0, on_failure="continue")
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

    # https://stackoverflow.com/a/21857841
    # Run test multiple times
    @pytest.mark.parametrize("execution_number", range(10))
    def test_reproducibility(self, execution_number):
        json_in = os.path.join(os.path.dirname(__file__), "data/test.json")
        match = mm.CaseMatchOneToMany.load(json_in)
        # Total is 36 so guaranteed to not hit all of them
        all_matched_pairs = match.create_matched_pairs(
            iterations=5, seed=63, n_jobs=2
        )
        match_df = all_matched_pairs.to_dataframe()

        exp_matrix = {
            0: ["S0B", "S0B", "S1B", "S1B", "S2B"],
            1: ["S1B", "S1B", "S2B", "S2B", "S1B"],
            2: ["S4B", "S4B", "S4B", "S4B", "S4B"],
            3: ["S5B", "S8B", "S5B", "S6B", "S7B"],
            4: ["S3B", "S3B", "S3B", "S3B", "S3B"],
            5: ["S8B", "S6B", "S8B", "S8B", "S8B"]
        }
        exp_df = pd.DataFrame(exp_matrix).T
        exp_df.index = [f"S{x}A" for x in range(6)]
        exp_df.index.name = "case_id"

        pd.testing.assert_frame_equal(exp_df, match_df)

    def test_on_failure_warn(self):
        s1 = pd.Series([0, 2, 3, 4, 5, 6, 7, 8, 100])
        s2 = pd.Series([3, 5, 7, 9, 4, 6, 6, 2, 50])
        s1.index = [f"S{x}A" for x in range(9)]
        s2.index = [f"S{x}B" for x in range(9)]

        with pytest.warns(UserWarning) as warn_info:
            match = match_by_single(s1, s2, 1.0, on_failure="warn")

        warn_msgs = set(map(lambda x: str(x.message), warn_info))
        assert len(warn_msgs) == 2

        exp_msg_1 = "No matches found for S8A"
        exp_msg_2 = "No matches found for S0A"

        assert set([exp_msg_1, exp_msg_2]) == warn_msgs

        exp_match = {
            "S0A": set(),
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


class TestCaseMatchCollection:
    def test_save_load(self, tmp_path):
        cases = [f"S{x+1}A" for x in range(5)]
        controls = [f"S{x+1}B" for x in range(10)]
        rng = np.random.default_rng()

        matches = [
            rng.choice(controls, size=len(cases), replace=False)
            for x in cases
        ]
        df = pd.DataFrame.from_records(matches, index=cases)
        df.index.name = "case_id"

        fpath_1 = f"{tmp_path}/coll_1.tsv"
        df.to_csv(fpath_1, sep="\t", index=True)
        df2 = mm.CaseMatchCollection.load(fpath_1).to_dataframe()
        pd.testing.assert_frame_equal(df, df2)

        cm_coll = [dict(df[x]) for x in df]
        cm_coll = mm.CaseMatchCollection([
            mm.CaseMatchOneToOne({k: {v} for k, v in cm.items()})
            for cm in cm_coll
        ])
        fpath_2 = f"{tmp_path}/coll_2.tsv"
        cm_coll.save(fpath_2)
        df3 = mm.CaseMatchCollection.load(fpath_2).to_dataframe()
        pd.testing.assert_frame_equal(df, df3)

    def test_apply(self):
        json_in = os.path.join(os.path.dirname(__file__), "data/test.json")
        match = mm.CaseMatchOneToMany.load(json_in)
        collection = match.create_matched_pairs(iterations=1000)

        rng = np.random.default_rng()
        cases = match.cases
        ctrls = match.controls

        case_vals = rng.normal(4, 0.25, size=len(cases))
        ctrl_vals = rng.normal(2, 0.25, size=len(ctrls))

        case_vals = pd.Series(case_vals, index=list(cases))
        ctrl_vals = pd.Series(ctrl_vals, index=list(ctrls))

        all_vals = pd.concat([case_vals, ctrl_vals])

        # Same ctrl set can be used in different arrangement of controls
        uniq_mean_sets = set()
        for cm in collection:
            this_set = frozenset(cm.controls)
            this_mean = np.mean(all_vals.loc[list(this_set)])
            uniq_mean_sets.add(this_mean)

        num_uniq_mean_sets = len(uniq_mean_sets)

        def cm_func(cm):
            _cases = list(cm.cases)
            _ctrls = list(cm.controls)

            _case_vals = all_vals.loc[_cases]
            _ctrl_vals = all_vals.loc[_ctrls]

            return np.mean(_case_vals), np.mean(_ctrl_vals)

        # All case vals means should be the same bc same cases each time
        gen = list(collection.apply(cm_func))
        case_means, ctrl_means = list(zip(*gen))

        assert len(set(case_means)) == 1
        assert len(set(ctrl_means)) == num_uniq_mean_sets
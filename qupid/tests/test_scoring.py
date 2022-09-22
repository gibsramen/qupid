from pkg_resources import resource_filename

import numpy as np
import pandas as pd
import pytest
import qupid

rng = np.random.default_rng(42)
asd_str = "Diagnosed by a medical professional (doctor, physician assistant)"
no_asd_str = "I do not have this condition"


@pytest.fixture
def scoring_metadata():
    metadata_fpath = resource_filename("qupid", "tests/data/asd.tsv")
    metadata = pd.read_table(metadata_fpath, sep="\t", index_col=0)
    n = metadata.shape[0]
    metadata["rand"] = rng.poisson(5, size=n)
    return metadata


@pytest.fixture
def scoring_case_match_mult(scoring_metadata):
    background = scoring_metadata.query("asd == @no_asd_str")
    focus = scoring_metadata.query("asd == @asd_str")

    cm = qupid.match_by_multiple(
        focus=focus,
        background=background,
        categories=["sex", "age_years", "rand"],
        tolerance_map={"age_years": 10, "rand": 3}
    )
    return cm


def test_cm_one_to_one_score(scoring_metadata, scoring_case_match_mult):
    cases = scoring_metadata[scoring_metadata["asd"] == asd_str].index
    ctrls = scoring_metadata[scoring_metadata["asd"] == no_asd_str].index

    collection = scoring_case_match_mult.create_matched_pairs(iterations=1)
    score_df = collection[0].evaluate_match_score(
        scoring_metadata, ["age_years", "rand"]
    )

    exp_cols = ["ctrl_id", "age_years_diff", "rand_diff"]
    assert (score_df.columns == exp_cols).all()
    assert score_df.index.name == "case_id"
    assert set(cases) == set(score_df.index)
    assert set(score_df["ctrl_id"]).issubset(set(ctrls))


def test_cm_collection_score(scoring_metadata, scoring_case_match_mult):
    collection = scoring_case_match_mult.create_matched_pairs(iterations=100)
    collection_df = collection.to_dataframe()
    score_df = collection.evaluate_match_scores(["age_years", "rand"])

    exp_cols = ["ctrl_id", "age_years_diff", "rand_diff", "match_num"]
    assert (score_df.columns == exp_cols).all()

    exp_match_nums = np.arange(100).astype(int)
    match_num_vc = score_df["match_num"].value_counts()
    assert (match_num_vc == 45).all()

    new_rng = np.random.default_rng()
    rand_match_num = new_rng.choice(exp_match_nums)

    single_match_score_df = score_df.query("match_num == @rand_match_num")
    single_match_cm = collection[rand_match_num]
    cases = single_match_cm.cases
    ctrls = single_match_cm.controls

    assert set(single_match_score_df.index) == set(cases)
    assert set(single_match_score_df["ctrl_id"]) == set(ctrls)

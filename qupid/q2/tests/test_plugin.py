from pkg_resources import resource_filename

import numpy as np
import pandas as pd
import pytest
from qiime2 import Artifact, Metadata
from qiime2.plugins import qupid
from skbio import DistanceMatrix

from qupid.q2._methods import match_one_to_many as raw_match_one_to_many

CASE_ID = "Diagnosed by a medical professional (doctor, physician assistant)"


@pytest.fixture(scope="module")
def metadata():
    fname = resource_filename("qupid", "tests/data/asd.tsv")
    md = pd.read_table(fname, sep="\t", index_col=0)
    return Metadata(md)


@pytest.fixture(scope="module")
def distance_matrix(metadata):
    md = metadata.to_dataframe()
    n = md.shape[0]
    rng = np.random.default_rng()

    dm = rng.beta(1, 1, (n, n))
    upper_tri = np.triu(dm, 1)
    dm = upper_tri + upper_tri.T
    dm = DistanceMatrix(dm, ids=list(md.index))
    return Artifact.import_data("DistanceMatrix", dm)


@pytest.fixture(scope="module")
def univariate(metadata):
    md = metadata.to_dataframe()
    n = md.shape[0]
    rng = np.random.default_rng()

    values = pd.Series(rng.gamma(1, 2, size=n), index=list(md.index))
    values.index.name = "sampleid"
    values.name = "faith_pd"
    values = pd.DataFrame(values)
    return Metadata(values)


def test_match_one_to_many(metadata):
    qupid.methods.match_one_to_many(
        sample_metadata=metadata,
        case_control_column="asd",
        categories=["sex", "age_years"],
        case_identifier=CASE_ID,
        tolerances=["age_years+-10"],
    )


def test_match_one_to_one(metadata):
    (cm_one_to_many,) = qupid.methods.match_one_to_many(
        sample_metadata=metadata,
        case_control_column="asd",
        categories=["sex", "age_years"],
        case_identifier=CASE_ID,
        tolerances=["age_years+-10"],
    )

    qupid.methods.match_one_to_one(
        case_match_one_to_many=cm_one_to_many,
        iterations=100,
    )


def test_shuffle(metadata):
    qupid.pipelines.shuffle(
        sample_metadata=metadata,
        case_control_column="asd",
        categories=["sex", "age_years"],
        case_identifier=CASE_ID,
        tolerances=["age_years+-10"],
        iterations=100,
    )


@pytest.fixture(scope="module")
def collection(metadata):
    _, coll = qupid.pipelines.shuffle(
        sample_metadata=metadata,
        case_control_column="asd",
        categories=["sex", "age_years"],
        case_identifier=CASE_ID,
        tolerances=["age_years+-10"],
        iterations=100,
    )
    return coll


def test_assessment_multivariate(collection, distance_matrix):
    qupid.visualizers.assess_matches_multivariate(
        case_match_collection=collection,
        distance_matrix=distance_matrix,
        permutations=999,
    )


def test_assessment_multivariate_correct(collection, distance_matrix):
    qupid.visualizers.assess_matches_multivariate(
        case_match_collection=collection,
        distance_matrix=distance_matrix,
        permutations=99,
        correct=True,
    )


@pytest.mark.parametrize("test", ["t", "mw", "paired-t", "wilcoxon"])
def test_assessment_univariate(collection, univariate, test):
    qupid.visualizers.assess_matches_univariate(
        case_match_collection=collection,
        data=univariate.get_column("faith_pd"),
        test=test,
    )


def test_assessment_univariate_correct(collection, univariate):
    qupid.visualizers.assess_matches_univariate(
        case_match_collection=collection,
        data=univariate.get_column("faith_pd"),
        correct=True,
    )


def test_assess_covariate_balance(metadata, collection):
    qupid.visualizers.assess_covariate_balance(
        case_match_collection=collection,
        sample_metadata=metadata,
        case_control_column="asd",
        case_identifier=CASE_ID,
        categories=["age_years"],
    )


@pytest.mark.parametrize(
    "bad_token,match_fragment",
    [
        ("age_years10", "Malformed tolerance token"),
        ("age_years+-+-5", "Malformed tolerance token"),
        ("age_years+-x", "Non-numeric tolerance value"),
        ("age_years+--5", "non-negative"),
    ],
)
def test_malformed_tolerance(metadata, bad_token, match_fragment):
    with pytest.raises(ValueError, match=match_fragment):
        raw_match_one_to_many(
            sample_metadata=metadata,
            case_control_column="asd",
            categories=["age_years"],
            case_identifier=CASE_ID,
            tolerances=[bad_token],
        )

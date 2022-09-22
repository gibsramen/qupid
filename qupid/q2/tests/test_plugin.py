from pkg_resources import resource_filename

import numpy as np
import pandas as pd
import pytest
from qiime2 import Artifact, Metadata
from qiime2.plugins import qupid
from skbio import DistanceMatrix


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
        case_identifier=(
            "Diagnosed by a medical professional (doctor, physician "
            "assistant)"
        ),
        tolerances=["age_years+-10"]
    )


def test_create_matched_pairs(metadata):
    cm_one_to_many, = qupid.methods.match_one_to_many(
        sample_metadata=metadata,
        case_control_column="asd",
        categories=["sex", "age_years"],
        case_identifier=(
            "Diagnosed by a medical professional (doctor, physician "
            "assistant)"
        ),
        tolerances=["age_years+-10"]
    )

    qupid.methods.create_matched_pairs(
        case_match_one_to_many=cm_one_to_many,
        iterations=100,
    )


def test_shuffle(metadata):
    qupid.pipelines.shuffle(
        sample_metadata=metadata,
        case_control_column="asd",
        categories=["sex", "age_years"],
        case_identifier=(
            "Diagnosed by a medical professional (doctor, physician "
            "assistant)"
        ),
        tolerances=["age_years+-10"],
        iterations=100,
    )


def test_assessment_multivariate(metadata, distance_matrix):
    _, coll = qupid.pipelines.shuffle(
        sample_metadata=metadata,
        case_control_column="asd",
        categories=["sex", "age_years"],
        case_identifier=(
            "Diagnosed by a medical professional (doctor, physician "
            "assistant)"
        ),
        tolerances=["age_years+-10"],
        iterations=100,
    )

    qupid.visualizers.assess_matches_multivariate(
        case_match_collection=coll,
        distance_matrix=distance_matrix,
        permutations=999
    )


def test_assessment_univariate(metadata, univariate):
    _, coll = qupid.pipelines.shuffle(
        sample_metadata=metadata,
        case_control_column="asd",
        categories=["sex", "age_years"],
        case_identifier=(
            "Diagnosed by a medical professional (doctor, physician "
            "assistant)"
        ),
        tolerances=["age_years+-10"],
        iterations=100,
    )

    qupid.visualizers.assess_matches_univariate(
        case_match_collection=coll,
        data=univariate.get_column("faith_pd"),
    )

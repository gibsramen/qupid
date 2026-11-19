import importlib

from qiime2.plugin import (
    Plugin,
    List,
    Str,
    Choices,
    Metadata,
    Bool,
    Int,
    MetadataColumn,
    Numeric,
)
from q2_types.distance_matrix import DistanceMatrix

from qupid import __version__
from qupid import _descriptions as DESC
from qupid.stats import _TEST_REGISTRY
from ._format import CaseMatchDirFmt, CaseMatchCollectionDirFmt
from ._type import CaseMatch, OneToMany, OneToOne, CaseMatchCollection
from ._methods import match_one_to_many, match_one_to_one
from ._visualizers import (
    assess_matches_multivariate,
    assess_matches_univariate,
    assess_covariate_balance,
)
from ._pipelines import shuffle


plugin = Plugin(
    name="qupid",
    version=__version__,
    website="https://github.com/gibsramen/qupid",
    short_description="Plugin for case-control matching",
    description=(
        "Match cases to controls based on metadata criteria for " "microbiome data."
    ),
    package="qupid",
)

MD_DESC = "Sample metadata for matching."
CC_COL_DESC = "Column in metadata to divide samples into cases and controls."
CATS_DESC = "Categories on which to match."
CASE_ID_DESC = "Identifier for cases in chosen column."
TOL_DESC = (
    "List of numeric columns and the tolerance within which to match. "
    "Use '+-' to separate column name from value (e.g. age_years+-10)."
)
FAIL_OPTS = DESC.VALID_ON_FAILURE_OPTS


plugin.methods.register_function(
    function=match_one_to_many,
    inputs={},
    input_descriptions={},
    parameters={
        "sample_metadata": Metadata,
        "case_control_column": Str,
        "categories": List[Str],
        "case_identifier": Str,
        "tolerances": List[Str],
        "on_failure": Str % Choices(set(FAIL_OPTS)),
    },
    parameter_descriptions={
        "sample_metadata": MD_DESC,
        "case_control_column": CC_COL_DESC,
        "categories": CATS_DESC,
        "case_identifier": CASE_ID_DESC,
        "tolerances": TOL_DESC,
        "on_failure": DESC.FAIL,
    },
    outputs=[("case_match_one_to_many", CaseMatch[OneToMany])],
    name="Match each case to all possible controls.",
    description=(
        "Creates a mapping of each case to all possible controls given "
        "the provided matching criteria. A control can be matched to multiple "
        "cases."
    ),
)

plugin.methods.register_function(
    function=match_one_to_one,
    inputs={"case_match_one_to_many": CaseMatch[OneToMany]},
    input_descriptions={"case_match_one_to_many": "Full mapping"},
    parameters={"iterations": Int, "strict": Bool, "seed": Int, "n_jobs": Int},
    parameter_descriptions={
        "iterations": DESC.ITERATIONS,
        "strict": DESC.STRICT,
        "seed": DESC.SEED,
        "n_jobs": DESC.JOBS,
    },
    outputs=[("case_match_collection", CaseMatchCollection)],
    name="Match each case to one control for multiple iterations.",
    description=(
        "Match each case to exactly one valid control for multiple "
        "iterations. A control can only be matched to a single case. "
        "Each generated matching is a unique mapping of cases to controls. "
        "Note that depending on the structure of the possible matches, "
        "there may be fewer valid matchings than iterations."
    ),
)

plugin.pipelines.register_function(
    function=shuffle,
    inputs={},
    input_descriptions={},
    parameters={
        "sample_metadata": Metadata,
        "case_control_column": Str,
        "categories": List[Str],
        "case_identifier": Str,
        "tolerances": List[Str],
        "on_match_failure": Str % Choices(set(FAIL_OPTS)),
        "iterations": Int,
        "strict": Bool,
        "seed": Int,
        "n_jobs": Int,
    },
    parameter_descriptions={
        "sample_metadata": MD_DESC,
        "case_control_column": CC_COL_DESC,
        "categories": CATS_DESC,
        "case_identifier": CASE_ID_DESC,
        "on_match_failure": DESC.FAIL,
        "tolerances": TOL_DESC,
        "iterations": DESC.ITERATIONS,
        "strict": DESC.STRICT,
        "seed": DESC.SEED,
        "n_jobs": DESC.JOBS,
    },
    outputs=[
        ("case_match_one_to_many", CaseMatch[OneToMany]),
        ("case_match_collection", CaseMatchCollection),
    ],
    name=(
        "Create multiple one-to-one case-control matches given matching " "criteria."
    ),
    description=(
        "Pipeline to get all valid controls per case and perform multiple "
        "iterations of matching each case to a single control."
    ),
)

_TEST_ALIASES = Str % Choices(set(_TEST_REGISTRY.keys()))

plugin.visualizers.register_function(
    function=assess_matches_multivariate,
    inputs={
        "case_match_collection": CaseMatchCollection,
        "distance_matrix": DistanceMatrix,
    },
    input_descriptions={
        "case_match_collection": "Iterations of one-to-one matchings.",
        "distance_matrix": "Distance matrix with all cases and controls.",
    },
    parameters={"permutations": Int, "n_jobs": Int, "correct": Bool},
    parameter_descriptions={
        "permutations": "Number of PERMANOVA permutations.",
        "n_jobs": DESC.JOBS,
        "correct": DESC.CORRECT,
    },
    name="Run PERMANOVA on all one-to-one matches.",
    description=(
        "Plot the distribution of p-values from PERMANOVA on all one-to-one "
        "matches (case vs. control).  Pass correct=True to append a "
        "Benjamini-Hochberg q_value column to the downloadable TSV."
    ),
)

plugin.visualizers.register_function(
    function=assess_matches_univariate,
    inputs={
        "case_match_collection": CaseMatchCollection,
    },
    input_descriptions={
        "case_match_collection": "Iterations of one-to-one matchings.",
    },
    parameters={
        "data": MetadataColumn[Numeric],
        "test": _TEST_ALIASES,
        "n_jobs": Int,
        "correct": Bool,
    },
    parameter_descriptions={
        "data": "Numeric data to assess matches.",
        "test": DESC.TEST,
        "n_jobs": DESC.JOBS,
        "correct": DESC.CORRECT,
    },
    name="Run a univariate test on all one-to-one matches.",
    description=(
        "Plot the distribution of p-values from a univariate test on all "
        "one-to-one matches (case vs. control).  Supports independent tests "
        "(t, Mann-Whitney) and paired tests (paired-t, Wilcoxon signed-rank) "
        "that exploit the matched-pair structure.  Pass correct=True to append "
        "a Benjamini-Hochberg q_value column to the downloadable TSV."
    ),
)

plugin.visualizers.register_function(
    function=assess_covariate_balance,
    inputs={
        "case_match_collection": CaseMatchCollection,
    },
    input_descriptions={
        "case_match_collection": "Iterations of one-to-one matchings.",
    },
    parameters={
        "sample_metadata": Metadata,
        "case_control_column": Str,
        "case_identifier": Str,
        "categories": List[Str],
    },
    parameter_descriptions={
        "sample_metadata": MD_DESC,
        "case_control_column": CC_COL_DESC,
        "case_identifier": CASE_ID_DESC,
        "categories": DESC.CATEGORIES,
    },
    name="Plot covariate balance (Love plot) pre and post matching.",
    description=(
        "Computes the standardized mean difference (SMD; Cohen's d) for each "
        "specified covariate before and after matching, averaged across all "
        "matchings in the collection, and renders a Love plot.  Values below "
        "|SMD| = 0.1 are considered well-balanced."
    ),
)


plugin.register_semantic_types(CaseMatch, OneToOne, OneToMany, CaseMatchCollection)
plugin.register_semantic_type_to_format(
    CaseMatch[OneToOne | OneToMany], artifact_format=CaseMatchDirFmt
)
plugin.register_semantic_type_to_format(
    CaseMatchCollection, artifact_format=CaseMatchCollectionDirFmt
)
plugin.register_formats(CaseMatchDirFmt, CaseMatchCollectionDirFmt)


importlib.import_module("qupid.q2._transformer")

import importlib

from qiime2.plugin import Plugin, List, Str, Choices, Metadata, Bool, Int
from qupid import __version__
from ._format import CaseMatchDirFmt, CaseMatchCollectionDirFmt
from ._type import CaseMatch, OneToMany, OneToOne, CaseMatchCollection
from ._methods import match_one_to_many, match_one_to_one


plugin = Plugin(
    name="qupid",
    version=__version__,
    website="https://github.com/gibsramen/qupid",
    short_description="Plugin for case-control matching",
    description=(
        "Match cases to controls based on metadata criteria for "
        "microbiome data."
    ),
    package="qupid"
)


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
        "on_failure": Str % Choices({"raise", "ignore"})
    },
    outputs=[("case_match_one_to_many", CaseMatch[OneToMany])],
    name="Match each case to all possible controls.",
    description="Pikachu"
)

plugin.methods.register_function(
    function=match_one_to_one,
    inputs={"case_match_one_to_many": CaseMatch[OneToMany]},
    input_descriptions={"case_match_one_to_many": "Full mapping"},
    parameters={
        "iterations": Int,
        "strict": Bool,
        "n_jobs": Int
    },
    outputs=[("case_match_collection", CaseMatchCollection)],
    name="Match each case to one control.",
    description="Raichu"
)

plugin.register_semantic_types(CaseMatch, OneToOne, OneToMany,
                               CaseMatchCollection)
plugin.register_semantic_type_to_format(
    CaseMatch[OneToOne | OneToMany],
    artifact_format=CaseMatchDirFmt
)
plugin.register_semantic_type_to_format(
    CaseMatchCollection,
    artifact_format=CaseMatchCollectionDirFmt
)
plugin.register_formats(CaseMatchDirFmt, CaseMatchCollectionDirFmt)


importlib.import_module("qupid.q2._transformer")

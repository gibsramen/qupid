import importlib

from qiime2.plugin import Plugin, List, Str, Choices, Metadata
from qupid import __version__
from ._format import CaseMatchOneToManyDirFmt
from ._type import CaseMatchOneToMany
from ._methods import match_one_to_many


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
    outputs=[("case_match_one_to_many", CaseMatchOneToMany)],
    name="Match one case to all possible controls.",
    description="Pikachu"
)

plugin.register_semantic_types(CaseMatchOneToMany)
plugin.register_semantic_type_to_format(
    CaseMatchOneToMany,
    artifact_format=CaseMatchOneToManyDirFmt
)
plugin.register_formats(CaseMatchOneToManyDirFmt)


importlib.import_module("qupid.q2._transformer")

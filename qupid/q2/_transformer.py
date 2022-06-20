import json

from .plugin_setup import plugin
from ._format import CaseMatchOneToManyFormat
from qupid.casematch import CaseMatchOneToMany


@plugin.register_transformer
def _1(data: CaseMatchOneToMany) -> CaseMatchOneToManyFormat:
    ff = CaseMatchOneToManyFormat()
    with ff.open() as fh:
        tmp_cc_map = {k: list(v) for k, v in data.case_control_map.items()}
        json.dump(tmp_cc_map, fh)
    return ff


@plugin.register_transformer
def _2(ff: CaseMatchOneToManyFormat) -> CaseMatchOneToMany:
    return CaseMatchOneToMany.load(str(ff))

import json

from .plugin_setup import plugin
from ._format import CaseMatchFormat
from qupid.casematch import CaseMatchOneToMany


@plugin.register_transformer
def _1(data: CaseMatchOneToMany) -> CaseMatchFormat:
    ff = CaseMatchFormat()
    with ff.open() as fh:
        tmp_cc_map = {k: list(v) for k, v in data.case_control_map.items()}
        json.dump(tmp_cc_map, fh)
    return ff


@plugin.register_transformer
def _2(ff: CaseMatchFormat) -> CaseMatchOneToMany:
    return CaseMatchOneToMany.load(str(ff))

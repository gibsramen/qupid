import json

import pandas as pd

from .plugin_setup import plugin
from ._format import CaseMatchFormat, CaseMatchCollectionFormat
from qupid.casematch import CaseMatchOneToMany, CaseMatchCollection


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


@plugin.register_transformer
def _3(data: CaseMatchCollection) -> CaseMatchCollectionFormat:
    ff = CaseMatchCollectionFormat()
    with ff.open() as fh:
        data.to_dataframe().to_csv(fh, sep="\t", index=True)
    return ff


@plugin.register_transformer
def _4(ff: CaseMatchCollectionFormat) -> CaseMatchCollection:
    return CaseMatchCollection.load(str(ff))


@plugin.register_transformer
def _5(data: pd.DataFrame) -> CaseMatchCollectionFormat:
    ff = CaseMatchCollectionFormat()
    with ff.open() as fh:
        data.to_csv(fh, sep="\t", index=True)
    return ff


@plugin.register_transformer
def _6(ff: CaseMatchCollectionFormat) -> pd.DataFrame:
    return pd.read_table(str(ff), sep="\t", index=True)

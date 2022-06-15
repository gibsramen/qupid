from typing import Iterator

import biom
import pandas as pd

from .casematch import CaseMatchOneToOne, CaseMatchCollection
from .decorators import check_input_types


@check_input_types(["table", "casematch"])
def filter_table(
    table: biom.Table,
    casematch: CaseMatchOneToOne,
) -> (biom.Table, pd.Series):
    """Filter table according to matched case-controls.

    :param table: Counts of features by samples
    :type table: biom.Table

    :param casematch: Matching of cases to controls
    :type casematch: qupid.CaseMatchOneToOne

    :returns: Filtered table and case-control mapping
    :rtype: (biom.Table, pd.Series)
    """
    cases = pd.Series("case", index=casematch.cases)
    ctrls = pd.Series("control", index=casematch.controls)
    case_ctrl = pd.concat([cases, ctrls])
    case_ctrl.name = "case_or_control"

    table_filt = table.filter(case_ctrl.index, inplace=False)
    return table_filt, case_ctrl


@check_input_types(["table", "collection"])
def filter_table_by_collection(
    table: biom.Table,
    collection: CaseMatchCollection,
) -> Iterator:
    """Filter table by multiple match sets.

    :param table: Counts of features by samples
    :type table: biom.Table

    :param collection: Collection of match sets
    :type collection: qupid.CaseMatchCollection

    :returns: Generator of filtered tables and mappings
    :rtype: Iterator
    """
    return (filter_table(table, cm) for cm in collection.case_matches)

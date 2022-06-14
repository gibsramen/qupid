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

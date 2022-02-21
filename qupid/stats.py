import pandas as pd
from skbio.stats.distance import permanova

from qupid.exceptions import NoDistanceMatrixError
from qupid.matching import CaseMatchOneToOne


def case_control_permanova(case_match: CaseMatchOneToOne,
                           permutations=999) -> pd.Series:
    """Run PERMANOVA on a CaseMatch instance.

    :param case_match: CaseMatch instance containing cases and controls (must
        contain DistanceMatrix)
    :type case_match: qupid.CaseMatchOneToOne

    :returns: Results of PERMANOVA between cases and controls (test statistic
        and p-value)
    :rtype: pd.Series
    """
    if case_match.distance_matrix is None:
        raise NoDistanceMatrixError()

    cases = list(case_match.cases)
    controls = list(case_match.controls)
    all_samples = cases + controls

    cc_list = ["case" for x in cases] + ["control" for x in controls]
    cc_df = pd.DataFrame.from_dict({"case_control": cc_list})
    cc_df.index = all_samples
    subset = case_match.distance_matrix.filter(all_samples)

    return permanova(subset, cc_df, "case_control", permutations)

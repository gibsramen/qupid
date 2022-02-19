import pandas as pd
import scipy.stats as ss
from skbio import DistanceMatrix
from skbio.stats.distance import permanova

from .exceptions import NoDistanceMatrixError
from .matching import CaseMatch


def case_control_permanova(case_match: CaseMatch) -> pd.Series:
    """Run PERMANOVA on a CaseMatch instance.

    :param case_match: CaseMatch instance containing cases and controls (must
        contain DistanceMatrix)
    :type case_match: qupid.CaseMatch

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
    return permanova(subset, cc_df, "case_control")


def run_stats(
    case_match: CaseMatch,
    dm: DistanceMatrix,
    iterations: int = 10
):
    results = []
    for i in range(iterations):
        greedy_df = case_match.greedy_match()
        subset = dm.filter(greedy_df.index)
        cases = greedy_df.query("case_or_control == 'case'").index
        controls = greedy_df.query("case_or_control == 'control'").index
        case_within = subset.within(cases)["value"].values
        ctrl_within = subset.within(controls)["value"].values
        btwn = subset.between(cases, controls)["value"]

        pnova_res = permanova(subset, greedy_df, "case_or_control")
        within_stat, within_p = ss.kruskal(case_within, ctrl_within)
        btwn_v_case_stat, btwn_v_case_p = ss.kruskal(btwn, case_within)
        btwn_v_ctrl_stat, btwn_v_ctrl_p = ss.kruskal(btwn, ctrl_within)
        results.append((
            pnova_res["test statistic"],
            pnova_res["p-value"],
            within_stat,
            within_p,
            btwn_v_case_stat,
            btwn_v_case_p,
            btwn_v_ctrl_stat,
            btwn_v_ctrl_p
        ))

    results_df = pd.DataFrame(
        results,
        columns=[
            "permanova stat", "permanova p", "within stat", "within p",
            "btwn and within case stat", "btwn and within case p",
            "btwn and within control stat", "btwn and within ctrl p"
        ]
    )
    return results_df

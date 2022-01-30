import pandas as pd
import scipy.stats as ss
from skbio import DistanceMatrix
from skbio.stats.distance import permanova

from .matching import CaseMatch


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

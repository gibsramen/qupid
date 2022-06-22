import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from skbio import DistanceMatrix

from qupid.casematch import CaseMatchCollection
from qupid import stats


def assess_matches_multivariate(
    output_dir: str,
    case_match_collection: pd.DataFrame,
    distance_matrix: DistanceMatrix,
    permutations: int = 999,
    n_jobs: int = 1,
) -> None:
    index_fp = os.path.join(output_dir, "index.html")
    fig_loc = os.path.join(output_dir, "permanova_pvalues.svg")
    fig_loc2 = os.path.join(output_dir, "permanova_pvalues.pdf")

    coll = CaseMatchCollection.from_dataframe(case_match_collection)
    print(distance_matrix.shape)

    pnova_df = stats.bulk_permanova(
        coll,
        distance_matrix,
        permutations,
        n_jobs,
    )

    results_loc = os.path.join(output_dir, "permanova_results.tsv")
    pnova_df.to_csv(results_loc, sep="\t", index=True)

    fig, ax = plt.subplots(1, 1, dpi=300, facecolor="white")
    sns.histplot(pnova_df["p-value"], ax=ax)
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.set_title(f"PERMANOVA p-values (n = {permutations})")

    plt.savefig(fig_loc)
    plt.savefig(fig_loc2)

    with open(index_fp, "w") as f:
        f.write("<html><body>\n")
        f.write("<font face='Arial'>\n")
        f.write(
            "<div style='text-align: center;'>\n"
            "<a href='permanova_pvalues.pdf' target='_blank'"
            "rel='noopener noreferrer'>"
            "Download plot as PDF</a><br>\n"
            "<a href='permanova_results.tsv'>Download results as TSV</a><br>\n"
        )
        f.write("<img src='permanova_pvalues.svg' alt='p-values'>\n")
        f.write("</div>\n")
        f.write("</font>")


def assess_matches_univariate(
    output_dir: str,
    case_match_collection: pd.DataFrame,
    data: pd.Series,
    n_jobs: int = 1,
) -> None:
    index_fp = os.path.join(output_dir, "index.html")
    fig_loc = os.path.join(output_dir, "univariate_pvalues.svg")
    fig_loc2 = os.path.join(output_dir, "univariate_pvalues.pdf")

    univariate_df = stats.bulk_univariate_test(
        CaseMatchCollection.from_dataframe(case_match_collection),
        data.to_dataframe().squeeze(),  # hack to deal with Q2 Metadata
        "t",
        n_jobs
    )

    results_loc = os.path.join(output_dir, "univariate_results.tsv")
    univariate_df.to_csv(results_loc, sep="\t", index=True)

    fig, ax = plt.subplots(1, 1, dpi=300, facecolor="white")
    sns.histplot(univariate_df["p-value"], ax=ax)
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.set_title("t-test p-values")

    plt.savefig(fig_loc)
    plt.savefig(fig_loc2)

    with open(index_fp, "w") as f:
        f.write("<html><body>\n")
        f.write("<font face='Arial'>\n")
        f.write(
            "<div style='text-align: center;'>\n"
            "<a href='univariate_pvalues.pdf' target='_blank'"
            "rel='noopener noreferrer'>"
            "Download plot as PDF</a><br>\n"
            "<a href='univariate_results.tsv'>"
            "Download results as TSV</a><br>\n"
        )
        f.write("<img src='univariate_pvalues.svg' alt='p-values'>\n")
        f.write("</div>\n")
        f.write("</font>")

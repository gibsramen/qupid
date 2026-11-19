import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from qiime2 import Metadata
from skbio import DistanceMatrix

from qupid import compute_covariate_balance
from qupid.casematch import CaseMatchCollection
from qupid import stats


def _write_index_html(
    output_dir: str,
    image_file: str,
    image_alt: str,
    pdf_file: str,
    tsv_file: str,
) -> None:
    """Write a standard visualizer index.html linking a plot, PDF, and TSV."""
    index_fp = os.path.join(output_dir, "index.html")
    with open(index_fp, "w") as f:
        f.write("<html><body>\n")
        f.write("<font face='Arial'>\n")
        f.write(
            "<div style='text-align: center;'>\n"
            f"<a href='{pdf_file}' target='_blank' rel='noopener noreferrer'>"
            "Download plot as PDF</a><br>\n"
            f"<a href='{tsv_file}'>Download results as TSV</a><br>\n"
        )
        f.write(f"<img src='{image_file}' alt='{image_alt}'>\n")
        f.write("</div>\n")
        f.write("</font>")


def assess_matches_multivariate(
    output_dir: str,
    case_match_collection: pd.DataFrame,
    distance_matrix: DistanceMatrix,
    permutations: int = 999,
    n_jobs: int = -1,
    correct: bool = False,
) -> None:
    fig_loc = os.path.join(output_dir, "permanova_pvalues.svg")
    fig_loc2 = os.path.join(output_dir, "permanova_pvalues.pdf")

    coll = CaseMatchCollection.from_dataframe(case_match_collection)

    pnova_df = stats.bulk_permanova(
        coll,
        distance_matrix,
        permutations,
        n_jobs,
        correct=correct,
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

    _write_index_html(
        output_dir,
        image_file="permanova_pvalues.svg",
        image_alt="p-values",
        pdf_file="permanova_pvalues.pdf",
        tsv_file="permanova_results.tsv",
    )


def assess_matches_univariate(
    output_dir: str,
    case_match_collection: pd.DataFrame,
    data: pd.Series,
    test: str = "t",
    n_jobs: int = -1,
    correct: bool = False,
) -> None:
    fig_loc = os.path.join(output_dir, "univariate_pvalues.svg")
    fig_loc2 = os.path.join(output_dir, "univariate_pvalues.pdf")

    univariate_df = stats.bulk_univariate_test(
        CaseMatchCollection.from_dataframe(case_match_collection),
        data.to_dataframe().squeeze(),  # hack to deal with Q2 Metadata
        test,
        n_jobs,
        correct=correct,
    )

    results_loc = os.path.join(output_dir, "univariate_results.tsv")
    univariate_df.to_csv(results_loc, sep="\t", index=True)

    method_name = univariate_df["method_name"].iloc[0] if len(univariate_df) else test
    fig, ax = plt.subplots(1, 1, dpi=300, facecolor="white")
    sns.histplot(univariate_df["p-value"], ax=ax)
    ax.set_xlabel("p-value")
    ax.set_ylabel("Count")
    ax.set_title(f"{method_name} p-values")

    plt.savefig(fig_loc)
    plt.savefig(fig_loc2)

    _write_index_html(
        output_dir,
        image_file="univariate_pvalues.svg",
        image_alt="p-values",
        pdf_file="univariate_pvalues.pdf",
        tsv_file="univariate_results.tsv",
    )


def assess_covariate_balance(
    output_dir: str,
    sample_metadata: Metadata,
    case_control_column: str,
    case_identifier: str,
    case_match_collection: pd.DataFrame,
    categories: list,
) -> None:
    """Render a Love plot of standardized mean differences pre/post matching.

    SMD values are averaged across all matchings in the collection.
    Numeric and boolean columns only; non-numeric columns are skipped.
    """
    md = sample_metadata.to_dataframe()
    focus = md[md[case_control_column] == case_identifier]
    background = md[md[case_control_column] != case_identifier]

    coll = CaseMatchCollection.from_dataframe(case_match_collection)

    all_balances = [
        compute_covariate_balance(focus, background, cm, list(categories))
        for cm in coll
    ]
    avg_balance = (
        pd.concat(all_balances)
        .groupby("covariate")[["smd_pre", "smd_post"]]
        .mean()
        .reset_index()
    )

    results_loc = os.path.join(output_dir, "covariate_balance.tsv")
    avg_balance.to_csv(results_loc, sep="\t", index=False)

    n = len(avg_balance)
    fig_h = max(3.0, n * 0.55 + 1.0)
    fig, ax = plt.subplots(1, 1, dpi=300, facecolor="white", figsize=(6, fig_h))

    y_pos = list(range(n))
    covariates = avg_balance["covariate"].tolist()

    ax.scatter(
        avg_balance["smd_pre"].abs(),
        y_pos,
        color="#CC3311",
        label="Pre-matching",
        zorder=3,
        s=50,
    )
    ax.scatter(
        avg_balance["smd_post"].abs(),
        y_pos,
        color="#4477AA",
        label="Post-matching",
        zorder=3,
        s=50,
    )
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.8, label="|SMD| = 0.1")
    ax.axvline(0.0, color="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(covariates)
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_title("Covariate Balance (Love Plot)")
    ax.legend(loc="lower right", fontsize=8)

    fig_loc = os.path.join(output_dir, "covariate_balance.svg")
    fig_loc2 = os.path.join(output_dir, "covariate_balance.pdf")
    plt.savefig(fig_loc, bbox_inches="tight")
    plt.savefig(fig_loc2, bbox_inches="tight")
    plt.close(fig)

    _write_index_html(
        output_dir,
        image_file="covariate_balance.svg",
        image_alt="Love plot",
        pdf_file="covariate_balance.pdf",
        tsv_file="covariate_balance.tsv",
    )

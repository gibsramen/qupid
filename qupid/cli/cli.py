import click
import pandas as pd
from skbio import DistanceMatrix

from qupid import (
    __version__,
    compute_covariate_balance,
    match_by_multiple,
    stats,
)
from qupid import shuffle as _shuffle
from qupid import _descriptions as DESC
from qupid.casematch import CaseMatchCollection
from qupid.stats import _TEST_REGISTRY


@click.group()
@click.version_option(__version__)
def qupid():
    """Performs case-control matching."""
    pass


_SHUFFLE_OPTIONS = [
    click.option("-f", "--focus", required=True, type=click.Path(), help=DESC.FOCUS),
    click.option(
        "-b", "--background", required=True, type=click.Path(), help=DESC.BACKGROUND
    ),
    click.option("-dc", "--discrete-cat", multiple=True, help=DESC.DC),
    click.option(
        "-nc", "--numeric-cat", multiple=True, type=(str, float), help=DESC.NC
    ),
    click.option(
        "--on-failure",
        default="raise",
        type=click.Choice(DESC.VALID_ON_FAILURE_OPTS, case_sensitive=False),
        help=DESC.FAIL,
        show_default=True,
    ),
    click.option(
        "--strict/--no-strict", default=True, help=DESC.STRICT, show_default=True
    ),
    click.option("-rs", "--random-seed", type=int, help=DESC.SEED, show_default=True),
    click.option("-j", "--jobs", type=int, help=DESC.JOBS, show_default=True),
]


def _add_options(options):
    """Decorator factory to attach a list of click options to a command."""

    def decorator(f):
        for opt in reversed(options):
            f = opt(f)
        return f

    return decorator


def _load_focus_background(focus_path, background_path):
    focus = pd.read_table(focus_path, sep="\t", index_col=0)
    background = pd.read_table(background_path, sep="\t", index_col=0)
    return focus, background


def _build_cats_and_tols(discrete_cat, numeric_cat):
    tol_map = {cat: float(tol) for cat, tol in numeric_cat}
    cats = list(tol_map.keys()) + list(discrete_cat)
    return cats, tol_map


@qupid.command()
@_add_options(_SHUFFLE_OPTIONS)
@click.option("-i", "--iterations", required=True, type=int, help=DESC.ITERATIONS)
@click.option("-o", "--output", type=click.Path(), required=True, help=DESC.OUTPUT)
def shuffle(
    focus,
    background,
    discrete_cat,
    numeric_cat,
    on_failure,
    strict,
    random_seed,
    jobs,
    iterations,
    output,
):
    """Create multiple 1:1 case-control matchings."""
    focus_df, background_df = _load_focus_background(focus, background)
    cats, tol_map = _build_cats_and_tols(discrete_cat, numeric_cat)

    res = _shuffle(
        focus=focus_df,
        background=background_df,
        categories=cats,
        tolerance_map=tol_map,
        iterations=iterations,
        on_failure=on_failure,
        strict=strict,
        seed=random_seed,
        n_jobs=jobs,
    )
    print(f"Created {res.shape[1]}/{iterations} match sets!")
    res.to_csv(output, sep="\t", index=True)


@qupid.command("match-groups")
@_add_options(_SHUFFLE_OPTIONS)
@click.option(
    "-nctrl",
    "--n-controls",
    required=True,
    type=int,
    help=DESC.N_CONTROLS,
)
@click.option("-i", "--iterations", required=True, type=int, help=DESC.ITERATIONS)
@click.option("-o", "--output", type=click.Path(), required=True, help=DESC.OUTPUT)
def match_groups(
    focus,
    background,
    discrete_cat,
    numeric_cat,
    on_failure,
    strict,
    random_seed,
    jobs,
    n_controls,
    iterations,
    output,
):
    """Generate k:1 matched groups (N distinct controls per case)."""
    focus_df, background_df = _load_focus_background(focus, background)
    cats, tol_map = _build_cats_and_tols(discrete_cat, numeric_cat)

    cm = match_by_multiple(
        focus=focus_df,
        background=background_df,
        categories=cats,
        tolerance_map=tol_map,
        on_failure=on_failure,
    )
    groups = cm.create_matched_groups(
        n_controls=n_controls,
        iterations=iterations,
        strict=strict,
        seed=random_seed,
        n_jobs=jobs,
    )

    rows = []
    for i, g in enumerate(groups):
        for case, ctrls in g.case_control_map.items():
            for ctrl in sorted(ctrls):
                rows.append({"iteration": i, "case": case, "control": ctrl})
    df = pd.DataFrame(rows, columns=["iteration", "case", "control"])
    df.to_csv(output, sep="\t", index=False)
    print(f"Created {len(groups)}/{iterations} match groups!")


@qupid.command("assess-univariate")
@click.option("-m", "--matches", required=True, type=click.Path(), help=DESC.MATCHES)
@click.option("-v", "--values", required=True, type=click.Path(), help=DESC.VALUES)
@click.option(
    "--test",
    default="t",
    type=click.Choice(sorted(_TEST_REGISTRY.keys()), case_sensitive=False),
    help=DESC.TEST,
    show_default=True,
)
@click.option("--correct", is_flag=True, default=False, help=DESC.CORRECT)
@click.option("-j", "--jobs", default=-1, type=int, help=DESC.JOBS, show_default=True)
@click.option("-o", "--output", type=click.Path(), required=True, help=DESC.OUTPUT)
def assess_univariate(matches, values, test, correct, jobs, output):
    """Run a univariate test across all case-control matchings."""
    coll = CaseMatchCollection.load(matches)
    vals_df = pd.read_table(values, sep="\t", index_col=0)
    vals = vals_df.iloc[:, 0]
    res = stats.bulk_univariate_test(
        coll, vals, test=test, correct=correct, n_jobs=jobs
    )
    res.to_csv(output, sep="\t", index=True)
    print(f"Results written to {output}")


@qupid.command("assess-multivariate")
@click.option("-m", "--matches", required=True, type=click.Path(), help=DESC.MATCHES)
@click.option(
    "-d",
    "--distance-matrix",
    required=True,
    type=click.Path(),
    help=DESC.DISTANCE_MATRIX,
)
@click.option(
    "-p",
    "--permutations",
    default=999,
    type=int,
    help=DESC.PERMUTATIONS,
    show_default=True,
)
@click.option("--correct", is_flag=True, default=False, help=DESC.CORRECT)
@click.option("-j", "--jobs", default=-1, type=int, help=DESC.JOBS, show_default=True)
@click.option("-o", "--output", type=click.Path(), required=True, help=DESC.OUTPUT)
def assess_multivariate(matches, distance_matrix, permutations, correct, jobs, output):
    """Run PERMANOVA across all case-control matchings."""
    coll = CaseMatchCollection.load(matches)
    dm = DistanceMatrix.read(distance_matrix)
    res = stats.bulk_permanova(
        coll, dm, permutations=permutations, correct=correct, n_jobs=jobs
    )
    res.to_csv(output, sep="\t", index=True)
    print(f"Results written to {output}")


@qupid.command("balance")
@click.option("-f", "--focus", required=True, type=click.Path(), help=DESC.FOCUS)
@click.option(
    "-b", "--background", required=True, type=click.Path(), help=DESC.BACKGROUND
)
@click.option(
    "-m",
    "--matches",
    required=True,
    type=click.Path(),
    help=(
        DESC.MATCHES + "  The first iteration column is used as the "
        "representative matching for post-matching SMD."
    ),
)
@click.option(
    "-c",
    "--category",
    "categories",
    multiple=True,
    required=True,
    help=DESC.CATEGORIES,
)
@click.option("-o", "--output", type=click.Path(), required=True, help=DESC.OUTPUT)
def balance(focus, background, matches, categories, output):
    """Report standardized mean difference (SMD) per covariate pre/post matching."""
    focus_df, background_df = _load_focus_background(focus, background)
    coll = CaseMatchCollection.load(matches)
    all_balances = [
        compute_covariate_balance(focus_df, background_df, cm, list(categories))
        for cm in coll
    ]
    res = (
        pd.concat(all_balances)
        .groupby("covariate")
        .mean(numeric_only=True)
        .reset_index()
    )
    res.to_csv(output, sep="\t", index=False)
    print(f"Balance report written to {output}")


if __name__ == "__main__":
    qupid()

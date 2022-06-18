import click
import pandas as pd

from qupid import __version__
from qupid import shuffle as _shuffle
import qupid._descriptions as DESC


@click.group()
@click.version_option(__version__)
def qupid():
    """Performs case-control matching."""
    pass


@qupid.command()
@click.option("-f", "--focus", required=True, type=click.Path(),
              help=DESC.FOCUS)
@click.option("-b", "--background", required=True, type=click.Path(),
              help=DESC.BACKGROUND)
@click.option("-i", "--iterations", required=True, type=int,
              help=DESC.ITERATIONS)
@click.option("-dc", "--discrete-cat", multiple=True, help=DESC.DC)
@click.option("-nc", "--numeric-cat", multiple=True, type=(str, float),
              help=DESC.NC)
@click.option("--raise-on-failure/--ignore-on-failure", default=True,
              help=DESC.FAIL)
@click.option("--strict/--no-strict", default=True, help=DESC.STRICT)
@click.option("-j", "--jobs", type=int, help=DESC.JOBS)
@click.option("-o", "--output", type=click.Path(), required=True,
              help=DESC.OUTPUT)
def shuffle(
    focus,
    background,
    iterations,
    discrete_cat,
    numeric_cat,
    raise_on_failure,
    strict,
    jobs,
    output
):
    tol_map = {cat: float(tol) for cat, tol in numeric_cat}

    numeric_cats = list(tol_map.keys())
    discrete_cats = list(discrete_cat)

    cats = numeric_cats + discrete_cats

    focus = pd.read_table(focus, sep="\t", index_col=0)
    background = pd.read_table(background, sep="\t", index_col=0)

    res = _shuffle(
        focus=focus,
        background=background,
        categories=cats,
        tolerance_map=tol_map,
        iterations=iterations,
        on_failure="raise" if raise_on_failure else "ignore",
        strict=strict,
        n_jobs=jobs
    )
    print(f"Created {res.shape[1]}/{iterations} match sets!")
    res.to_csv(output, sep="\t", index=True)


if __name__ == "__main__":
    qupid()

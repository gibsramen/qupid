import os
from pkg_resources import resource_filename

from click.testing import CliRunner
import numpy as np
import pandas as pd
from skbio import DistanceMatrix

from qupid.cli.cli import qupid

ASD_STR = "Diagnosed by a medical professional (doctor, physician assistant)"
NO_ASD_STR = "I do not have this condition"


def _load_split():
    """Return (focus, background) DataFrames from the bundled ASD metadata."""
    metadata_fpath = resource_filename("qupid", "tests/data/asd.tsv")
    metadata = pd.read_table(metadata_fpath, sep="\t", index_col=0)
    focus = metadata[metadata["asd"] == ASD_STR]
    background = metadata[metadata["asd"] == NO_ASD_STR]
    return focus, background


def _write_split(focus, background):
    """Write focus.tsv and background.tsv into the current directory."""
    focus.to_csv("focus.tsv", sep="\t", index=True)
    background.to_csv("background.tsv", sep="\t", index=True)


def _run_shuffle(runner, iterations, output="match.tsv"):
    """Build a match collection from focus.tsv/background.tsv via shuffle."""
    return runner.invoke(
        qupid,
        [
            "shuffle",
            "-f",
            "focus.tsv",
            "-b",
            "background.tsv",
            "-i",
            iterations,
            "-dc",
            "sex",
            "-nc",
            "age_years",
            10,
            "-o",
            output,
        ],
    )


def test_cli():
    runner = CliRunner()
    focus, background = _load_split()

    with runner.isolated_filesystem():
        _write_split(focus, background)

        result = _run_shuffle(runner, iterations=15)

        assert result.exit_code == 0, result.output
        assert os.path.exists("match.tsv")
        df = pd.read_table("match.tsv", sep="\t", index_col=0)
        assert df.shape == (45, 15)


def test_match_groups():
    runner = CliRunner()
    focus, background = _load_split()

    with runner.isolated_filesystem():
        _write_split(focus, background)

        result = runner.invoke(
            qupid,
            [
                "match-groups",
                "-f",
                "focus.tsv",
                "-b",
                "background.tsv",
                "-dc",
                "sex",
                "-nc",
                "age_years",
                10,
                "--n-controls",
                2,
                "-i",
                5,
                "-rs",
                42,
                "-o",
                "groups.tsv",
            ],
        )

        assert result.exit_code == 0, result.output
        assert os.path.exists("groups.tsv")
        df = pd.read_table("groups.tsv", sep="\t")
        assert list(df.columns) == ["iteration", "case", "control"]
        counts = df.groupby(["iteration", "case"])["control"].count()
        assert (counts == 2).all()


def test_assess_univariate():
    runner = CliRunner()
    focus, background = _load_split()
    all_samples = pd.concat([focus, background])
    ids = list(all_samples.index)

    with runner.isolated_filesystem():
        _write_split(focus, background)
        _run_shuffle(runner, iterations=10)

        rng = np.random.default_rng(0)
        vals = pd.Series(rng.gamma(2, size=len(ids)), index=ids, name="value")
        vals.to_csv("values.tsv", sep="\t", header=True)

        result = runner.invoke(
            qupid,
            [
                "assess-univariate",
                "-m",
                "match.tsv",
                "-v",
                "values.tsv",
                "--test",
                "paired-t",
                "--correct",
                "-o",
                "uni_results.tsv",
            ],
        )

        assert result.exit_code == 0, result.output
        res = pd.read_table("uni_results.tsv", sep="\t", index_col=0)
        assert "q_value" in res.columns
        assert "p-value" in res.columns


def test_assess_multivariate():
    runner = CliRunner()
    focus, background = _load_split()
    all_samples = pd.concat([focus, background])
    ids = list(all_samples.index)
    n = len(ids)

    with runner.isolated_filesystem():
        _write_split(focus, background)
        _run_shuffle(runner, iterations=5)

        rng = np.random.default_rng(1)
        raw = rng.beta(1, 1, (n, n))
        dm_arr = np.triu(raw, 1) + np.triu(raw, 1).T
        dm = DistanceMatrix(dm_arr, ids=ids)
        dm.write("dm.tsv")

        result = runner.invoke(
            qupid,
            [
                "assess-multivariate",
                "-m",
                "match.tsv",
                "-d",
                "dm.tsv",
                "-p",
                99,
                "--correct",
                "-o",
                "mv_results.tsv",
            ],
        )

        assert result.exit_code == 0, result.output
        res = pd.read_table("mv_results.tsv", sep="\t", index_col=0)
        assert "q_value" in res.columns


def test_balance():
    runner = CliRunner()
    focus, background = _load_split()

    with runner.isolated_filesystem():
        _write_split(focus, background)
        _run_shuffle(runner, iterations=5)

        result = runner.invoke(
            qupid,
            [
                "balance",
                "-f",
                "focus.tsv",
                "-b",
                "background.tsv",
                "-m",
                "match.tsv",
                "-c",
                "age_years",
                "-o",
                "balance.tsv",
            ],
        )

        assert result.exit_code == 0, result.output
        res = pd.read_table("balance.tsv", sep="\t")
        assert list(res.columns) == ["covariate", "smd_pre", "smd_post"]
        assert "age_years" in res["covariate"].values

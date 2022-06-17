import os
from pkg_resources import resource_filename

from click.testing import CliRunner
import pandas as pd

from qupid.cli.cli import qupid


def test_cli():
    runner = CliRunner()

    metadata_fpath = resource_filename("qupid", "tests/data/asd.tsv")
    metadata = pd.read_table(metadata_fpath, sep="\t", index_col=0)

    asd_str = (
        "Diagnosed by a medical professional (doctor, physician "
        "assistant)"
    )
    no_asd_str = "I do not have this condition"

    background = metadata.query("asd == @no_asd_str")
    focus = metadata.query("asd == @asd_str")

    bg_file = "background.tsv"
    focus_file = "focus.tsv"
    match_file = "match.tsv"

    with runner.isolated_filesystem():
        background.to_csv(bg_file, sep="\t", index=True)
        focus.to_csv(focus_file, sep="\t", index=True)

        result = runner.invoke(qupid, [
            "shuffle",
            "-f", focus_file,
            "-b", bg_file,
            "-i", 15,
            "-dc", "sex",
            "-nc", "age_years", 10,
            "-o", match_file
        ])

        assert result.exit_code == 0
        assert os.path.exists(match_file)

        df = pd.read_table(match_file, sep="\t", index_col=0)
        assert df.shape == (45, 15)

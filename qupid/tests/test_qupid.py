from pkg_resources import resource_filename

import pandas as pd

from qupid import shuffle


def test_shuffle():
    metadata_fpath = resource_filename("qupid", "tests/data/asd.tsv")
    metadata = pd.read_table(metadata_fpath, sep="\t", index_col=0)

    # Designate focus samples
    asd_str = "Diagnosed by a medical professional (doctor, physician assistant)"
    no_asd_str = "I do not have this condition"

    background = metadata.query("asd == @no_asd_str")
    focus = metadata.query("asd == @asd_str")

    res = shuffle(
        focus=focus,
        background=background,
        categories=["sex", "age_years"],
        tolerance_map={"age_years": 10},
        iterations=100
    )
    assert res.shape == (45, 100)

    matches = {tuple(res[i]) for i in res}
    assert len(matches) == 100

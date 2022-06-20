import pandas as pd
from qiime2 import Metadata

from qupid.casematch import CaseMatchOneToMany
from qupid import match_by_multiple


def match_one_to_many(
    sample_metadata: Metadata,
    case_control_column: str,
    categories: list,
    case_identifier: str,
    tolerances: list = None,
    on_failure: str = "raise"
) -> CaseMatchOneToMany:
    sample_metadata = sample_metadata.to_dataframe()
    focus = sample_metadata[
        sample_metadata[case_control_column] == case_identifier
    ]
    background = sample_metadata[
        sample_metadata[case_control_column] != case_identifier
    ]

    # age_years+-5 bmi+-3.0
    tolerance_map = dict([x.split("+-") for x in tolerances])
    tolerance_map = {k: float(v) for k, v in tolerance_map.items()}

    cm_one_to_many = match_by_multiple(
        focus=focus,
        background=background,
        categories=categories,
        tolerance_map=tolerance_map,
        on_failure=on_failure
    )
    return cm_one_to_many


def match_one_to_one(
    case_match_one_to_many: CaseMatchOneToMany,
    iterations: int = 10,
    strict: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    res = case_match_one_to_many.create_matched_pairs(
        iterations=iterations,
        strict=strict,
        n_jobs=n_jobs
    )
    return res.to_dataframe()

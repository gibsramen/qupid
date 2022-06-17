import numpy as np
import pandas as pd
import pytest

import qupid._casematch_utils as util


class TestMatchers:
    def test_match_continuous(self):
        focus_value = 1.0
        background_values = np.array([1.0, 2.0, 0.1, 0.5, 2.1, -0.1])
        tol = 1.0
        exp_hits = np.array([True, True, True, True, False, False])

        hits = util._match_continuous(focus_value, background_values, tol)
        assert (exp_hits == hits).all()

    def test_match_discrete(self):
        focus_value = "a"
        background_values = np.array(["a", "b", "c", "a", "a"])
        exp_hits = np.array([True, False, False, True, True])

        hits = util._match_discrete(focus_value, background_values)
        assert (exp_hits == hits).all()


def test_infer_types():
    a = pd.Series([1, 2, 3, 4, 5])
    b = pd.Series(["A", "B", "C", "D", "E"])

    cat1 = util._infer_column_type(a, a)
    assert cat1 == "continuous"

    cat2 = util._infer_column_type(b, b)
    assert cat2 == "discrete"

    with pytest.raises(ValueError) as exc_info:
        util._infer_column_type(a, b)

    exp_err_msg = "Focus and background do not have the same dtype"
    assert exp_err_msg == str(exc_info.value)

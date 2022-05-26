[![GitHub Actions CI](https://github.com/gibsramen/qupid/actions/workflows/main.yml/badge.svg)](https://github.com/gibsramen/qupid/actions)
[![PyPI](https://img.shields.io/pypi/v/qupid.svg)](https://pypi.org/project/qupid)

# qupid

(Pronounced like cupid)

qupid is a tool for generating and statisticall evualuating *multiple* case-control matchings of microbiome data.

## Installation

You can install the most up-to-date version of qupid from PyPi using the following command:

```
pip install qupid
```

## Usage

There are three primary steps to the qupid workflow:

1. Match each case to all valid controls
2. Generate multiple one-to-one matchings
3. Evaluate the statistical differences between cases and controls for all matchings

To match each case to all valid controls, we need to first establish matching criteria.
qupid allows matching by both categorical metadata (exact matches) and continuous metadata (matching within provided tolerance).
You can match on either a single metadata column or based on multiple.

In qupid, the cases to be matched are referred to as the "focus" set, while the set of all possible controls is called the "background".
For this tutorial we will be used data from the American Gut Project to match cases to controls in samples from people with autism.

First, we'll load in the provided example metadata and separate it into the focus (samples from people with autism) and the background (samples from people who do not have autism).

```python
from pkg_resources import resource_filename
import pandas as pd

metadata_fpath = resource_filename("qupid", "tests/data/asd.tsv")
metadata = pd.read_table(metadata_fpath, sep="\t", index_col=0)

# Designate focus samples
asd_str = "Diagnosed by a medical professional (doctor, physician assistant)"
background = metadata.query("asd != @asd_str")
focus = metadata.query("asd == @asd_str")
```

Next, we want to perform case-control matching on sex and age.
Sex is a discrete factor, so qupid will attempt to find exact matches (e.g. male to male, female to female).
However, age is a continuous factor; as a result, we should provide a tolerance value (e.g. match within 10 years).
We use the `match_by_multiple` function to match based on more than one metadata category.

```python
from qupid import match_by_multiple

cm = match_by_multiple(
    focus=focus,
    background=background,
    category_type_map={"sex": "discrete", "age_years": "continuous"},
    tolerance_map={"age_years": 10}
)
```

This creates a `CaseMatchOneToMany` object where each case is matched to each possible control.
You can view the underlying matches as a dictionary with `cm.case_control_map`.

What we now want is to match each case to a *single* control so we can perform downstream analysis.
However, we have *a lot* of possible controls.
We can easily see how many cases and possible controls we have.

```python
print(len(cm.cases), len(cm.controls))
```

This tells us that we have 45 cases and 1785 possible controls.
Because of this, there are many possible sets of valid matchings of each case to a single control.
We can use qupid to generate many such cases.

```python
results = cm.create_matched_pairs(iterations=100)
```

This creates a `CaseMatchCollection` data structure that contains 100 `CaseMatchOneToOne` instances.
Each `CaseMatchOneToOne` entry maps each case to *a single control* rather than all possible controls.
We can verify that each entry has exactly 45 cases and 45 controls.

```python
print(len(results[0].cases), len(results[0].controls))
```

qupid provides a convenience method to convert a `CaseMatchCollection` object into a pandas DataFrame.
The DataFrame index corresponds to the cases, while each column represents a distinct set of matching controls.
The value in a cell represents a matching control to the row's case.

```python
results_df = results.to_dataframe()
results_df.head()
```

```
                                0                 1   ...                98                99
case_id                                               ...
S10317.000026181  S10317.000033804  S10317.000069086  ...  S10317.000108605  S10317.000076381
S10317.000071491  S10317.000155409  S10317.000103912  ...  S10317.000099277  S10317.000036401
S10317.000029293  S10317.000069676  S10317.X00175749  ...  S10317.000069299  S10317.000066846
S10317.000067638  S10317.X00179103  S10317.000052409  ...  S10317.000067511  S10317.000067601
S10317.000067637  S10317.000067747  S10317.000098161  ...  S10317.000017116  S10317.000067997

[5 rows x 100 columns]
```

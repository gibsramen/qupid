[![Main CI](https://github.com/gibsramen/qupid/actions/workflows/main.yml/badge.svg)](https://github.com/gibsramen/qupid/actions)
[![QIIME 2 CI](https://github.com/gibsramen/qupid/actions/workflows/q2.yml/badge.svg)](https://github.com/gibsramen/qupid/actions/workflows/q2.yml)
[![PyPI](https://img.shields.io/pypi/v/qupid.svg)](https://pypi.org/project/qupid)

# Qupid

(Pronounced like cupid)

Qupid is a tool for generating and statistically evaluating *multiple* case-control matchings of microbiome data.

## Installation

You can install the most up-to-date version of Qupid from PyPi using the following command:

```
pip install qupid
```

## Quickstart

Qupid provides a convenience function, `shuffle`, to easily generate multiple matches based on matching critiera.
This block of code will determine each viable control per case and randomly pick 10 arrangments of a single case matched to a single valid control.
The output is a pandas DataFrame where the rows are case names and each column represents a valid mapping of case to control.

```python
from pkg_resources import resource_filename
import pandas as pd
import qupid

metadata_fpath = resource_filename("qupid", "tests/data/asd.tsv")
metadata = pd.read_table(metadata_fpath, sep="\t", index_col=0)

asd_str = "Diagnosed by a medical professional (doctor, physician assistant)"
no_asd_str = "I do not have this condition"

background = metadata.query("asd == @no_asd_str")
focus = metadata.query("asd == @asd_str")

matches = qupid.shuffle(
    focus=focus,
    background=background,
    categories=["sex", "age_years"],
    tolerance_map={"age_years": 10},
    iterations=100
)
```

## Tutorial

There are three primary steps to the Qupid workflow:

1. Match each case to all valid controls
2. Generate multiple one-to-one matchings
3. Evaluate the statistical differences between cases and controls for all matchings

To match each case to all valid controls, we need to first establish matching criteria.
Qupid allows matching by both categorical metadata (exact matches) and continuous metadata (matching within provided tolerance).
You can match on either a single metadata column or based on multiple.

In Qupid, the cases to be matched are referred to as the "focus" set, while the set of all possible controls is called the "background".
For this tutorial we will be used data from the American Gut Project to match cases to controls in samples from people with autism.

First, we'll load in the provided example metadata and separate it into the focus (samples from people with autism) and the background (samples from people who do not have autism).

### Loading data

```python
from pkg_resources import resource_filename
import pandas as pd

metadata_fpath = resource_filename("qupid", "tests/data/asd.tsv")
metadata = pd.read_table(metadata_fpath, sep="\t", index_col=0)

# Designate focus samples
asd_str = "Diagnosed by a medical professional (doctor, physician assistant)"
no_asd_str = "I do not have this condition"

background = metadata.query("asd == @no_asd_str")
focus = metadata.query("asd == @asd_str")
```

### Matching each case to all possible controls

Next, we want to perform case-control matching on sex and age.
Sex is a discrete factor, so Qupid will attempt to find exact matches (e.g. male to male, female to female).
However, age is a continuous factor; as a result, we should provide a tolerance value (e.g. match within 10 years).
We use the `match_by_multiple` function to match based on more than one metadata category.

```python
from qupid import match_by_multiple

cm = match_by_multiple(
    focus=focus,
    background=background,
    categories=["sex", "age_years"],
    tolerance_map={"age_years": 10}
)
```

This creates a `CaseMatchOneToMany` object where each case is matched to each possible control.
You can view the underlying matches as a dictionary with `cm.case_control_map`.

### Generating mappings from each case to a single control

What we now want is to match each case to a *single* control so we can perform downstream analysis.
However, we have *a lot* of possible controls.
We can easily see how many cases and possible controls we have.

```python
print(len(cm.cases), len(cm.controls))
```

This tells us that we have 45 cases and 1785 possible controls.
Because of this, there are many possible sets of valid matchings of each case to a single control.
We can use Qupid to generate many such cases.

```python
results = cm.create_matched_pairs(iterations=100, seed=42)
```

This creates a `CaseMatchCollection` data structure that contains 100 `CaseMatchOneToOne` instances.
Each `CaseMatchOneToOne` entry maps each case to *a single control* rather than all possible controls.
We can verify that each entry has exactly 45 cases and 45 controls.

```python
print(len(results[0].cases), len(results[0].controls))
```

Qupid provides a convenience method to convert a `CaseMatchCollection` object into a pandas DataFrame.
The DataFrame index corresponds to the cases, while each column represents a distinct set of matching controls.
The value in a cell represents a matching control to the row's case.

```python
results_df = results.to_dataframe()
results_df.head()
```

```
                                0                 1   ...                98                99
case_id                                               ...
S10317.000014115  S10317.000022626  S10317.000026613  ...  S10317.X00179815  S10317.X00185470
S10317.000015573  S10317.000067569  S10317.000102715  ...  S10317.X00179598  S10317.000053311
S10317.000020752  S10317.000072372  S10317.000103626  ...  S10317.000031341  S10317.000022109
S10317.000021552  S10317.000084542  S10317.000031594  ...  S10317.000108624  S10317.000033677
S10317.000021558  S10317.000113466  S10317.000109876  ...  S10317.000036071  S10317.000065484

[5 rows x 100 columns]
```

### Statistical assessment of matchings

Once we have this list of matchings, we want to determine how statistically difference cases are from controls based on some values.
Qupid supports two types of statistical tests: univariate and multivariate.
Univariate data is in the form of a vector where each case and control has a single value.
This can be alpha diversity, log-ratios, etc.
Multivariate data is in the form of a distance matrix where each entry is the pairwise distance between two samples, e.g. from beta diversity analysis.
We will generate random data for this tutorial where there exists a small difference between ASD samples and non-ASD samples.

```python
import numpy as np

rng = np.random.default_rng(42)
asd_mean = 4
ctrl_mean = 3.75

num_cases = len(cm.cases)
num_ctrls = len(cm.controls)

asd_values = rng.normal(asd_mean, 1, size=num_cases)
ctrl_values = rng.normal(ctrl_mean, 1, size=num_ctrls)

asd_values = pd.Series(asd_values, index=focus.index)
ctrl_values = pd.Series(ctrl_values, index=background.index)

sample_values = pd.concat([asd_values, ctrl_values])
```

We can now evaluate a t-test between case values and control values for each possible case-control matching in our collection.

```python
from qupid.stats import bulk_univariate_test

test_results = bulk_univariate_test(
    casematches=results,
    values=sample_values,
    test="t"
)
test_results.head()
```

This returns a DataFrame of test results sorted by descending test statistic.

```
   method_name test_statistic_name  test_statistic   p-value  sample_size  number_of_groups
58      t-test                   t        3.950109  0.000157           90                 2
4       t-test                   t        3.667203  0.000419           90                 2
11      t-test                   t        3.228335  0.001750           90                 2
57      t-test                   t        3.217517  0.001810           90                 2
17      t-test                   t        3.086428  0.002709           90                 2
..         ...                 ...             ...       ...          ...               ...
19      t-test                   t        0.294450  0.769107           90                 2
47      t-test                   t        0.227653  0.820444           90                 2
52      t-test                   t        0.176479  0.860323           90                 2
2       t-test                   t       -0.303799  0.761998           90                 2
46      t-test                   t       -0.752030  0.454040           90                 2

[100 rows x 6 columns]
```

From this table, we can see that iteration 58 best separates cases from controls based on our random data.
Conversely, iteration 34 showed essentially no difference between cases and controls.
This shows that it is important to create multiple matchings as some of them are better than others.
We can plot the distribution of p-values to get a sense of the overall distribution.

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(test_results["p-value"])
```

![p-value Histogram](https://raw.githubusercontent.com/gibsramen/qupid/main/imgs/asd_match_pvals.png)

We see that most of the p-values are near zero which makes sense because we simulated our data with a difference between ASD and non-ASD samples.

### Evaluating match scores

When providing continuous matching criteria, you may be interested in the closeness of cases and matched controls based on these differences.
Currently, Qupid treats all matches within a tolerance as equal.
However, you can evaluate the matching score of each set of matches for all your continuous categories.

```python
results_score_df = results.evaluate_match_scores(["age_years"])
results_score_df.head()
```

```
            case_id           ctrl_id age_years_diff match_num
0  S10317.000014115  S10317.000022626            0.0         0
1  S10317.000015573  S10317.000067569           -2.0         0
2  S10317.000020752  S10317.000072372           -9.0         0
3  S10317.000021552  S10317.000084542           -4.0         0
4  S10317.000021558  S10317.000113466            1.0         0
```

We see that for each case-ctrl match, the difference in `age_years` (case minus control).
Each row also includes the match iteration in which it appears for potential grouping analyses.

### Saving and loading qupid results

Qupid allows the saving and loading of both `CaseMatch` and `CaseMatchCollection` objects.
`CaseMatchOneToMany` and `CaseMatchOneToOne` objects are saved as JSON files while `CaseMatchCollection` objects are saved as pandas DataFrames.

```python
from qupid.casematch import CaseMatchOneToMany, CaseMatchOneToOne, CaseMatchCollection

cm.save("asd_matches.one_to_many.json")  # Save all possible matches
results.save("asd_matches.100.tsv")  # Save all 100 iterations
results[15].save("asd_matches.best.json")  # Save best matching

CaseMatchOneToMany.load("asd_matches.one_to_many.json")
CaseMatchCollection.load("asd_matches.100.tsv")
CaseMatchOneToOne.load("asd_matches.best.json")
```

## Command Line Interface

Qupid has a command line interface to create multiple matchings from cases and possible controls.
If providing numeric categories, the column name must be accompanied by the tolerance after a space (e.g. `age_years 5` for a tolerance of 5 years).
You can pass multiple options to `--discrete-cat` or `--numeric-cat` to specify multiple matching criteria.

For usage detalls, use `qupid shuffle --help`.

```
qupid shuffle \
    --focus focus.tsv \
    --background background.tsv \
    --iterations 15 \
    --discrete-cat sex \
    --discrete-cat race \
    --numeric-cat age_years 5 \
    --numeric-cat weight_lbs 10 \
    --output matches.tsv
```

## QIIME 2 Usage

Qupid provides support for the popular QIIME 2 framework of microbiome data analysis.
We assume in this tutorial that you are familiar with using QIIME 2 on the command line.
If not, we recommend you read the excellent [documentation](https://docs.qiime2.org/) before you get started with Qupid.

Run `qiime qupid --help` to see all possible commands.

### Matching one-to-many

Use `qiime qupid match-one-to-many` to match each case to all possible controls.
Note that for numeric categories, you must pass in tolerances in the form of `<column_name>+-<tolerance>`.

```
qiime qupid match-one-to-many \
    --m-sample-metadata-file metadata.tsv \
    --p-case-control-column case_control \
    --p-categories sex age_years \
    --p-case-identifier case \
    --p-tolerances age_years+-10 \
    --o-case-match-one-to-many cm_one_to_many.qza
```

### Matching one-to-one

With a one-to-many match, you can generate multiple possible one-to-one matches using `qiime qupid match-one-to-one`.

```
qiime qupid match-one-to-one \
    --i-case-match-one-to-many cm_one_to_many.qza \
    --p-iterations 10 \
    --o-case-match-collection cm_collection.qza
```

### Qupid shuffle

The previous two commands can be run sequentially using `qiime qupid shuffle`.

```
qiime qupid shuffle \
    --m-sample-metadata-file metadata.tsv \
    --p-case-control-column case_control \
    --p-categories sex age_years \
    --p-case-identifier case \
    --p-tolerances age_years+-10 \
    --p-iterations 10 \
    --output-dir shuffle
```

### Statistical assessment of matches

You can assess how different cases are from controls using both univariate data (such as alpha diversity) or multivariate data (distance matrices).
The result will be a histogram of p-values from either a t-test (univariate) or PERMANOVA (multivariate) comparing cases to controls.
Note that for either command, the input data must contain values for all possible cases and controls.

```
qiime qupid assess-matches-univariate \
    --i-case-match-collection cm_collection.qza \
    --m-data-file data.tsv \
    --m-data-column faith_pd \
    --o-visualization univariate_p_values.qzv
```

```
qiime qupid assess-matches-multivariate \
    --i-case-match-collection cm_collection.qza \
    --i-distance-matrix uw_unifrac_distance_matrix.qza \
    --p-permutations 999 \
    --o-visualization multivariate_p_values.qzv
```

## Help with Qupid

If you encounter a bug in Qupid, please post a GitHub issue and we will get to it as soon as we can. We welcome any ideas or documentation updates/fixes so please submit an issue and/or a pull request if you have thoughts on making Qupid better.

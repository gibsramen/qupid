VALID_ON_FAILURE_OPTS = ["raise", "warn", "continue"]

FOCUS = "Case samples to be matched."
BACKGROUND = "Possible control samples to match to cases."
ITERATIONS = "Number of matching iterations to perform."
DC = "Discrete category column name."
NC = "Numeric category column name and tolerance."
FAIL = (
    "Whether to 'raise' an error, 'warn', or 'continue' (silently) when "
    "no matches are found for a focus sample."
)
STRICT = (
    "Whether to perform strict matching such that all cases must be "
    "matched to a control."
)
JOBS = "Number of CPUs to use for parallelization."
SEED = "Integer to use as random seed."
OUTPUT = "Path to save matches."

MATCHES = "Path to a case-match TSV produced by 'shuffle' or 'match-groups'."
VALUES = "Path to a two-column TSV (sample_id, value) for univariate testing."
DISTANCE_MATRIX = (
    "Path to a square, tab-separated distance matrix with a header row and "
    "index column of sample IDs."
)
TEST = (
    "Statistical test to use.  Independent: 't' / 'ttest' / 't-test', "
    "'mw' / 'mann-whitney'.  Paired: 'paired-t' / 'ttest-rel', "
    "'wilcoxon' / 'wilcoxon-sr'."
)
CORRECT = "If set, append a Benjamini-Hochberg FDR 'q_value' column to the results."
PERMUTATIONS = "Number of PERMANOVA permutations."
CATEGORIES = "Metadata column(s) to include in the covariate-balance report."

N_CONTROLS = "Number of distinct controls to assign per case (k in k:1 matching)."

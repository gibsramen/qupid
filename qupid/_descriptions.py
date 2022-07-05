VALID_ON_FAILURE_OPTS = ["raise", "warn", "continue"]

FOCUS = "Case samples to be matched."
BACKGROUND = "Possible control samples to match to cases."
ITERATIONS = "Number of matching iterations to perform."
DC = "Discrete category column name."
NC = "Numeric category column name and tolerance."
FAIL = ("Whether to 'raise' an error, 'warn', or 'continue' (silently) when "
        "no matches are found for a focus sample.")
STRICT = ("Whether to perform strict matching such that all cases must be "
          "matched to a control.")
JOBS = "Number of CPUs to use for parallelization."
OUTPUT = "Path to save matches."

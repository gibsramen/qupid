def shuffle(
    ctx,
    sample_metadata,
    case_control_column,
    categories,
    case_identifier,
    tolerances=None,
    on_match_failure="raise",
    iterations=10,
    strict=True,
    seed=None,
    n_jobs=1,
):
    match_one_to_many = ctx.get_action("qupid", "match_one_to_many")
    create_matched_pairs = ctx.get_action("qupid", "create_matched_pairs")

    results = []
    cm_one_to_many, = match_one_to_many(
        sample_metadata=sample_metadata,
        case_control_column=case_control_column,
        categories=categories,
        case_identifier=case_identifier,
        tolerances=tolerances,
        on_failure=on_match_failure
    )
    results.append(cm_one_to_many)

    cm_collection, = create_matched_pairs(
        case_match_one_to_many=cm_one_to_many,
        iterations=iterations,
        strict=strict,
        seed=seed,
        n_jobs=n_jobs
    )
    results.append(cm_collection)

    return tuple(results)

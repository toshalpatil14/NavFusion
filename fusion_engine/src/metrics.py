import math


def calculate_position_error(
    true_x,
    true_y,
    estimated_x,
    estimated_y,
):
    """
    Calculate Euclidean position error in metres.
    """
    return math.sqrt(
        (estimated_x - true_x) ** 2
        + (estimated_y - true_y) ** 2
    )


def calculate_metrics(
    true_x_values,
    true_y_values,
    estimated_x_values,
    estimated_y_values,
    blackout_mask=None,
):
    """
    Calculate navigation performance metrics.

    blackout_mask:
        Optional list of True/False values.
        True means that sample belongs to GNSS blackout.
    """

    errors = []

    for true_x, true_y, estimated_x, estimated_y in zip(
        true_x_values,
        true_y_values,
        estimated_x_values,
        estimated_y_values,
    ):

        error = calculate_position_error(
            true_x,
            true_y,
            estimated_x,
            estimated_y,
        )

        errors.append(error)

    if not errors:
        raise ValueError("No trajectory data provided.")

    mean_error = sum(errors) / len(errors)

    max_error = max(errors)

    rmse = math.sqrt(
        sum(error ** 2 for error in errors)
        / len(errors)
    )

    # ------------------------------------------
    # BLACKOUT-SPECIFIC METRICS
    # ------------------------------------------

    blackout_errors = []
    blackout_distance = 0.0

    if blackout_mask is not None:

        for i in range(len(errors)):

            if blackout_mask[i]:
                blackout_errors.append(errors[i])

                # Calculate travelled distance
                # between consecutive blackout points
                if i > 0 and blackout_mask[i - 1]:

                    dx = (
                        true_x_values[i]
                        - true_x_values[i - 1]
                    )

                    dy = (
                        true_y_values[i]
                        - true_y_values[i - 1]
                    )

                    blackout_distance += math.sqrt(
                        dx ** 2 + dy ** 2
                    )

    # Default values if no blackout exists
    if blackout_errors:

        max_blackout_error = max(blackout_errors)
        final_blackout_error = blackout_errors[-1]

        if blackout_distance > 0:

            drift_percentage = (
                max_blackout_error
                / blackout_distance
            ) * 100

        else:
            drift_percentage = 0.0

    else:

        max_blackout_error = 0.0
        final_blackout_error = 0.0
        drift_percentage = 0.0

    # ------------------------------------------
    # SIH REQUIREMENT CHECK
    # ------------------------------------------

    sih_requirement_percent = 10.0

    passed = drift_percentage < sih_requirement_percent

    return {
        "mean_error_m": mean_error,
        "max_error_m": max_error,
        "rmse_m": rmse,
        "blackout_distance_m": blackout_distance,
        "max_blackout_error_m": max_blackout_error,
        "final_blackout_error_m": final_blackout_error,
        "drift_percentage": drift_percentage,
        "sih_requirement_percent": sih_requirement_percent,
        "passed": passed,
    }


def print_metrics_report(metrics):
    """
    Print a readable navigation performance report.
    """

    print("\n=== NAVIGATION PERFORMANCE REPORT ===\n")

    print(
        f"Mean Position Error:       "
        f"{metrics['mean_error_m']:.3f} m"
    )

    print(
        f"Maximum Position Error:    "
        f"{metrics['max_error_m']:.3f} m"
    )

    print(
        f"RMSE:                      "
        f"{metrics['rmse_m']:.3f} m"
    )

    print()

    print(
        f"GNSS Blackout Distance:    "
        f"{metrics['blackout_distance_m']:.3f} m"
    )

    print(
        f"Max Blackout Error:        "
        f"{metrics['max_blackout_error_m']:.3f} m"
    )

    print(
        f"Final Blackout Error:      "
        f"{metrics['final_blackout_error_m']:.3f} m"
    )

    print(
        f"Drift Percentage:          "
        f"{metrics['drift_percentage']:.3f}%"
    )

    print()

    print(
        f"SIH Requirement:           "
        f"< {metrics['sih_requirement_percent']:.1f}%"
    )

    if metrics["passed"]:
        print("RESULT: PASS ✓")
    else:
        print("RESULT: FAIL ✗")

    print()
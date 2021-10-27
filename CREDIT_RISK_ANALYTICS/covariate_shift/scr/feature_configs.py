import tensorflow_lattice as tfl
import numpy as np

# benötigt train_xs, feature_name_indices


def compute_quantiles(
    features, num_keypoints=10, clip_min=None, clip_max=None, missing_value=None
):
    # Clip min and max if desired.
    if clip_min is not None:
        features = np.maximum(features, clip_min)
        features = np.append(features, clip_min)
    if clip_max is not None:
        features = np.minimum(features, clip_max)
        features = np.append(features, clip_max)
    # Make features unique.
    unique_features = np.unique(features)
    # Remove missing values if specified.
    if missing_value is not None:
        unique_features = np.delete(
            unique_features, np.where(unique_features == missing_value)
        )
    # Compute and return quantiles over unique non-missing feature values.
    return np.quantile(
        unique_features,
        np.linspace(0.0, 1.0, num=num_keypoints),
        interpolation="nearest",
    ).astype(float)


# hier fängt die fun an; input:  feature_names = feature_names, train_xs = train_xy$train_xs

# assets, income u solche vars noch mal überdenken; eig müssten die alle decreasing sein


def generate_fconfigs(df_name, train_xs, feature_names):
    # Create a
    feature_name_indices = {name: index for index, name in enumerate(feature_names)}
    # Feature configs are used to specify how each feature is calibrated and used.
    if df_name == "ger":
        feature_config = [
            tfl.configs.FeatureConfig(
                name="status",
                monotonicity="decreasing",
                # We must set the keypoints manually.
                pwl_calibration_num_keypoints=4,
                pwl_calibration_input_keypoints=np.linspace(
                    np.min(train_xs[feature_name_indices["status"]]),
                    np.max(train_xs[feature_name_indices["status"]]),
                    num=4,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="duration",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=3,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["duration"]],
                    num_keypoints=3,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="credit_history",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=np.linspace(
                    np.min(train_xs[feature_name_indices["credit_history"]]),
                    np.max(train_xs[feature_name_indices["credit_history"]]),
                    num=5,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="purpose",
                num_buckets=10,
            ),
            tfl.configs.FeatureConfig(
                name="amount",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=10,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["amount"]], num_keypoints=10
                ),
            ),
            tfl.configs.FeatureConfig(
                name="savings",
                monotonicity="decreasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=np.linspace(
                    np.min(train_xs[feature_name_indices["savings"]]),
                    np.max(train_xs[feature_name_indices["savings"]]),
                    num=5,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="installment_rate",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=4,
                pwl_calibration_input_keypoints=np.linspace(
                    np.min(train_xs[feature_name_indices["installment_rate"]]),
                    np.max(train_xs[feature_name_indices["installment_rate"]]),
                    num=4,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="other_debtors",
                num_buckets=3,
            ),
            tfl.configs.FeatureConfig(
                name="property",
                num_buckets=4,
            ),
            tfl.configs.FeatureConfig(
                name="age",
                lattice_size=3,
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["age"]],
                    num_keypoints=5,
                    clip_max=100,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="other_installment_plans",
                num_buckets=3,
            ),
            tfl.configs.FeatureConfig(
                name="housing",
                num_buckets=3,
            ),
            tfl.configs.FeatureConfig(
                name="number_credits",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=4,
                pwl_calibration_input_keypoints=np.linspace(
                    np.min(train_xs[feature_name_indices["number_credits"]]),
                    np.max(train_xs[feature_name_indices["number_credits"]]),
                    num=4,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="job",
                num_buckets=4,
            ),
            tfl.configs.FeatureConfig(
                name="people_liable",
                num_buckets=2,
            ),
        ]
    elif df_name == "gmc":
        feature_config = [
            tfl.configs.FeatureConfig(
                name="unsecure_lines",
                monotonicity="increasing",
                # We must set the keypoints manually.
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["unsecure_lines"]],
                    num_keypoints=5,  # , clip_max=10 # Mit clipmax ließen sich die ausreißer rauskicken
                ),
                # Per feature regularization.
                # regularizer_configs=[
                #    tfl.configs.RegularizerConfig(name='calib_wrinkle', l2=0.1),
                # ],
            ),
            tfl.configs.FeatureConfig(
                name="age",
                lattice_size=3,
                monotonicity="increasing",
                # We must set the keypoints manually.
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["age"]], num_keypoints=5, clip_max=109
                ),
                # Per feature regularization.
                # regularizer_configs=[
                #    tfl.configs.RegularizerConfig(name='calib_wrinkle', l2=0.1),
                # ],
            ),
            tfl.configs.FeatureConfig(
                name="nr_past_due30",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["nr_past_due30"]],
                    num_keypoints=5,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="debt_ratio",
                monotonicity="increasing",
                # Keypoints that are uniformly spaced.
                pwl_calibration_num_keypoints=10,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["debt_ratio"]], num_keypoints=10
                ),
            ),
            tfl.configs.FeatureConfig(
                name="monthly_income",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=10,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["monthly_income"]], num_keypoints=10
                ),
            ),
            tfl.configs.FeatureConfig(
                name="nr_open_credits",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["nr_open_credits"]],
                    num_keypoints=5,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="nr_90days_late",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["nr_90days_late"]], num_keypoints=5
                ),
            ),
            tfl.configs.FeatureConfig(
                name="nr_re_loans",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["nr_re_loans"]],
                    num_keypoints=5,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="nr_past_due60",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["nr_past_due60"]],
                    num_keypoints=5,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="nr_dependents",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["nr_dependents"]], num_keypoints=5
                ),
            ),
        ]
    elif df_name == "pak":
        feature_config = [
            tfl.configs.FeatureConfig(
                name="sex",
                num_buckets=3,
            ),
            tfl.configs.FeatureConfig(
                name="marriage",
                num_buckets=8,
            ),
            tfl.configs.FeatureConfig(
                name="nr_dependants",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=np.linspace(
                    np.min(train_xs[feature_name_indices["nr_dependants"]]),
                    np.max(train_xs[feature_name_indices["nr_dependants"]]),
                    num=5,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="income",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["income"]],
                    num_keypoints=5,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="income_other",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["income_other"]],
                    num_keypoints=5,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="bank_accounts",
                num_buckets=3,
            ),
            tfl.configs.FeatureConfig(
                name="bank_accounts_sp",
                num_buckets=3,
            ),
            tfl.configs.FeatureConfig(
                name="assets",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["assets"]],
                    num_keypoints=5,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="age",
                lattice_size=3,
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["age"]],
                    num_keypoints=5,
                    clip_max=100,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="credit_card",
                monotonicity=[(0, 1)],
                num_buckets=2,
            ),
        ]
    elif df_name == "tcd":
        feature_config = [
            tfl.configs.FeatureConfig(
                name="limit_bal",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["limit_bal"]],
                    num_keypoints=5,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="sex",
                monotonicity="none",
                num_buckets=2,
            ),
            tfl.configs.FeatureConfig(
                name="education",
                monotonicity="none",
                num_buckets=7,
            ),
            tfl.configs.FeatureConfig(
                name="age",
                lattice_size=3,
                monotonicity="increasing",
                pwl_calibration_num_keypoints=5,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["age"]],
                    num_keypoints=5,
                    clip_max=100,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="pay_0",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=11,
                pwl_calibration_input_keypoints=np.linspace(
                    np.min(train_xs[feature_name_indices["pay_0"]]),
                    np.max(train_xs[feature_name_indices["pay_0"]]),
                    num=11,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="pay_2",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=11,
                pwl_calibration_input_keypoints=np.linspace(
                    np.min(train_xs[feature_name_indices["pay_2"]]),
                    np.max(train_xs[feature_name_indices["pay_2"]]),
                    num=11,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="pay_3",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=11,
                pwl_calibration_input_keypoints=np.linspace(
                    np.min(train_xs[feature_name_indices["pay_3"]]),
                    np.max(train_xs[feature_name_indices["pay_3"]]),
                    num=11,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="pay_4",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=11,
                pwl_calibration_input_keypoints=np.linspace(
                    np.min(train_xs[feature_name_indices["pay_4"]]),
                    np.max(train_xs[feature_name_indices["pay_4"]]),
                    num=11,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="pay_5",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=10,
                pwl_calibration_input_keypoints=np.linspace(
                    np.min(train_xs[feature_name_indices["pay_5"]]),
                    np.max(train_xs[feature_name_indices["pay_5"]]),
                    num=10,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="pay_6",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=10,
                pwl_calibration_input_keypoints=np.linspace(
                    np.min(train_xs[feature_name_indices["pay_6"]]),
                    np.max(train_xs[feature_name_indices["pay_6"]]),
                    num=10,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="pay_amt1",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=10,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["pay_amt1"]],
                    num_keypoints=10,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="pay_amt2",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=10,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["pay_amt2"]],
                    num_keypoints=10,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="pay_amt3",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=10,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["pay_amt3"]],
                    num_keypoints=10,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="pay_amt4",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=10,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["pay_amt4"]],
                    num_keypoints=10,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="pay_amt5",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=10,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["pay_amt5"]],
                    num_keypoints=10,
                ),
            ),
            tfl.configs.FeatureConfig(
                name="pay_amt6",
                monotonicity="increasing",
                pwl_calibration_num_keypoints=10,
                pwl_calibration_input_keypoints=compute_quantiles(
                    train_xs[feature_name_indices["pay_amt6"]],
                    num_keypoints=10,
                ),
            ),
        ]

    return feature_config
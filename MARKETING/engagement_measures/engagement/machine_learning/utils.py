import pandas as pd
import numpy as np
import shap

from typing import Union

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from xgboost import XGBClassifier
from matplotlib import pyplot
from alibi.explainers.ale import ALE, plot_ale


def _dummy_model(X_train: pd.DataFrame, y_train: pd.Series) -> DummyClassifier:
    """
    Build a dummy model in order to compare ml model with no skill model
    """
    model_dummy = DummyClassifier(strategy='stratified')
    model_dummy.fit(X_train, y_train)
    return model_dummy


def _feature_contribution(
        shap_values: np.ndarray,
        X_train: pd.DataFrame,
        model_name: str
) -> pd.DataFrame:
    """
    1. Save a summary plot for the SHAP values showing negative and positive contribution of features
    3. Return a df with features and corresponding share of contribution
    """
    shap.summary_plot(shap_values, X_train.values, feature_names=X_train.columns, show=False)
    pyplot.savefig(
        "engagement/machine_learning/figures/{}_summary_plot.png".format(model_name),
        format="png",
        dpi=150,
        bbox_inches='tight'
    )
    pyplot.close()
    column_list = X_train.columns
    feature_ratio = (np.abs(shap_values).sum(0) / np.abs(shap_values).sum()) * 100
    column_list_all = column_list[np.argsort(feature_ratio)[::-1]]
    feature_ratio_order_all = np.sort(feature_ratio)[::-1]
    contribution_df = pd.DataFrame(data={'feature': column_list_all, 'ratio': feature_ratio_order_all})
    return contribution_df


def _dummies_to_initial_cat(
        df: pd.DataFrame,
        genre: list,
        day_of_week: list,
        day_time: list
) -> pd.DataFrame:
    """
    In order to not have separated features for the initial categorical features, the dummy variables are summarized
    to their initial group (genre, dayofweek, daytime)
    """
    genre_contribution = (
        df['ratio'][df['feature']
                    .isin(genre)]
        .sum()
    )

    dayofweek_contribution = (
        df['ratio'][df['feature']
                    .isin(day_of_week)]
        .sum()
    )

    daytime_contribution = (
        df['ratio'][df['feature']
                    .isin(day_time)]
        .sum()
    )

    genre_df = pd.DataFrame(data={'feature': ['genre'], 'ratio': [genre_contribution]})
    dayofweek_df = pd.DataFrame(data={'feature': ['day_of_week'], 'ratio': [dayofweek_contribution]})
    daytime_df = pd.DataFrame(data={'feature': ['day_time'], 'ratio': [daytime_contribution]})

    df = df[~df['feature'].isin(day_of_week + genre + day_time)]
    features_contribution = (
        pd.concat(
            [genre_df, df, dayofweek_df, daytime_df]
        )
        .sort_values(
            by=['ratio'],
            ascending=False
        )
    )

    return features_contribution


def _compare_engagement_non_engagement(df: pd.DataFrame, engagement: list, non_engagement: list) -> pd.DataFrame:
    """
    When compare_engage_non_engage is set to 'True' contributions of engagement features and contributions
    of non engagement features are summarized to compare both groups
    """
    engagement_contribution = (
        df['ratio'][df['feature']
                    .isin(engagement)]
        .sum()
    )

    non_engagement_contribution = (
        df['ratio'][df['feature']
                    .isin(non_engagement)]
        .sum()
    )
    engagement_df = pd.DataFrame(data={'feature': ['engagement'], 'ratio': [engagement_contribution]})
    non_engagement_df = pd.DataFrame(data={'feature': ['non_engagement'], 'ratio': [non_engagement_contribution]})
    contribution_df = pd.concat([engagement_df, non_engagement_df]).sort_values(by=['ratio'], ascending=False)
    return contribution_df


def _build_ale_plots(
        model: Union[LogisticRegression, KerasClassifier, RandomForestClassifier, XGBClassifier],
        X_train: pd.DataFrame,
        mc_columns: list,
        model_name: str
) -> None:
    """
    In order to check effects of multicollinearity on SHAP values the collinear features are visualized through ALE
    plots
    """
    proba = model.predict_proba
    mc_columns_index = [X_train.columns.get_loc(c) for c in mc_columns if c in X_train]
    proba_ale = ALE(proba, feature_names=X_train.columns, target_names=[0, 1])
    proba_exp = proba_ale.explain(X_train.values, features=mc_columns_index)
    plot_ale(proba_exp, n_cols=2, fig_kw={'figwidth': 30, 'figheight': 30})
    pyplot.savefig(
        "engagement/machine_learning/figures/{}_ale_plots.png".format(model_name),
        format="png",
        dpi=150,
        bbox_inches='tight'
    )
    pyplot.close()
    return None

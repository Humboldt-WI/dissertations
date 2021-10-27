import pandas as pd
import numpy as np
import pickle
import shap

from typing import Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from IPython.display import display

from config import config
from utils import (
    _dummy_model, _feature_contribution,
    _build_ale_plots, _dummies_to_initial_cat,
    _compare_engagement_non_engagement
)


def glm_model(
        feature_list: list,
        model_name: str,
        compare_engage_non_engage: bool = False,
        load_model: bool = False
) -> pd.DataFrame:
    """
    1.  Building a Logistic Regression Model
    2.  Check performance (roc-auc, pr-auc) and compare it with dummy classifier
    3.  Calculate SHAP values
    4.1 When 'compare_engage_non_engage' is set to 'False' contribution of individual engagement features are
        calculated and compared. Furthermore, ALE is calculated for the collinear features and the corresponding
        Plots are saved
    4.2 When 'compare_engage_non_engage' is set to 'True' engagement and non-engagement features are summarized
        and the contribution to the overall prediction is compared between both groups
    5.  When 'load_model' is set to 'True' an existing model is loaded and used for further process otherwise
        a new model is trained
    """
    X_train, X_test, y_train, y_test = _load_train_test('train_prepared', 'test_prepared', feature_list, 'orders')
    glm = _build_save_glm(X_train, y_train, model_name, load_model=load_model)
    dummy_model = _dummy_model(X_train, y_train)
    _model_performance(glm, dummy_model, X_test, y_test)
    shap_values = _calculate_shap(glm, X_train)
    contribution_df = _feature_contribution(shap_values, X_train, model_name)

    if compare_engage_non_engage:
        contribution_df = _compare_engagement_non_engagement(
            contribution_df,
            config['engagement_features'],
            config['non_engagement_features']
        )
    else:
        contribution_df = _dummies_to_initial_cat(
            contribution_df,
            config['genre'],
            config['day_of_week'],
            config['day_time']
        )
        _build_ale_plots(glm, X_train, config['mc_columns'], model_name)

    return contribution_df


def _load_train_test(
        train_name: str,
        test_name: str,
        X_columns: list,
        y_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Load train and test set and split it up to sets containing features only and target variable only
    """
    train = pd.read_parquet('engagement/machine_learning/data/{}.parquet'.format(train_name))
    test = pd.read_parquet('engagement/machine_learning/data/{}.parquet'.format(test_name))
    X_train = train[X_columns]
    y_train = train[y_column]
    X_test = test[X_columns]
    y_test = test[y_column]
    return X_train, X_test, y_train, y_test


def _build_save_glm(
        X_train: pd.DataFrame, y_train: pd.Series, model_name: str, load_model: bool = False,
        penalty: str = "l2", C: int = 1.0, solver: str = 'liblinear'
) -> LogisticRegression:
    """
    If 'load_model' is true the model with the defined name is loaded otherwise a new glm is built and saved
    """
    if load_model:
        model = pickle.load(open('engagement/machine_learning/models/{}.sav'.format(model_name), 'rb'))
    else:
        model = LogisticRegression(penalty=penalty, C=C, solver=solver)
        model.fit(X_train, y_train)
        pickle.dump(model, open('engagement/machine_learning/models/{}.sav'.format(model_name), 'wb'))
    return model


def _model_performance(
        model: LogisticRegression,
        dummy_model: DummyClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> None:
    """
    Check the performance (roc_auc, pr_auc) of the model and compare it with random classifier
    """
    glm_pred = model.predict_proba(X_test)[:, 1]
    dummy_pred = dummy_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, glm_pred)
    roc_auc_dummy = roc_auc_score(y_test, dummy_pred)
    glm_precision, glm_recall, _ = precision_recall_curve(y_test, glm_pred)
    glm_pr_auc = auc(glm_recall, glm_precision)
    dummy_precision, dummy_recall, _ = precision_recall_curve(y_test, dummy_pred)
    dummy_pr_auc = auc(dummy_recall, dummy_precision)
    print('GLM ROC AUC: %.3f' % roc_auc)
    print('NO SKILL ROC AUC: %.3f' % roc_auc_dummy)
    print('GLM PR AUC: %.3f' % glm_pr_auc)
    print('NO SKILL PR AUC: %.3f' % dummy_pr_auc)
    return None


def _calculate_shap(model: LogisticRegression, X_train: pd.DataFrame) -> np.ndarray:
    """
    Calculate SHAP values for further procedure
    """
    shap.initjs()
    explainer = shap.LinearExplainer(model, X_train, feature_perturbation='correlation_dependent')
    shap_values = explainer.shap_values(X_train)
    return shap_values


if __name__ == "__main__":
    """
    Function is executed for engagement features only and for all features
    """
    contribution_engagement = glm_model(
        config['engagement_features'],
        'glm_engagement_features',
        compare_engage_non_engage=False,
        load_model=True
    )
    contribution_all_features = glm_model(
        config['all_features'],
        'glm_all_features',
        compare_engage_non_engage=True,
        load_model=True
    )
    display(contribution_engagement)
    display(contribution_all_features)

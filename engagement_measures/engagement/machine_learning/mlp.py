import pandas as pd
import numpy as np
import shap
import tensorflow as tf

from typing import Tuple

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split
from IPython.display import display

from config import config
from utils import (
    _dummy_model, _feature_contribution,
    _build_ale_plots, _dummies_to_initial_cat,
    _compare_engagement_non_engagement
)


def mlp_model(
        feature_list: list, model_name: str,
        compare_engage_non_engage: bool = False,
        load_model: bool = False
) -> pd.DataFrame:
    """
    1.  Build Multi Layer Perceptron Neural Network
    1.1 Two models with the same structure are built. The first one is used for performance check and
        calculation of SHAP values and the second model is used for ALE. Reason is that SHAP package is only
        compatible with Sequential type and ALE module is only compatible with KerasClassifier.
    2.  Check performance (roc-auc, pr-auc) and compare it with dummy classifier
    3.  Calculate SHAP values
    4.1 When 'compare_engage_non_engage' is set to 'False' contribution of individual engagement features are
        calculated and compared. Furthermore, ALE is calculated for the collinear features and the corresponding
        plots are saved
    4.2 When 'compare_engage_non_engage' is set to 'True' engagement and non-engagement features are summarized
        and the contribution to the overall prediction is compared between both groups
    5.  When 'load_model' is set to 'True' an existing model is loaded and used for further process otherwise
        a new model is trained (model for ALE is always trained fresh as KerasClassifier type models do not
        have a save method)
    """
    X_train, X_valid, X_test, y_train, y_valid, y_test = _load_train_test(
        'train_prepared',
        'test_prepared',
        feature_list,
        'orders'
    )
    mlp_shap = _model_for_shap(X_train, y_train, X_valid, y_valid, model_name, load_model=load_model)
    mlp_ale = _model_for_ale(X_train, y_train, X_valid, y_valid)
    dummy_model = _dummy_model(X_train, y_train)
    _model_performance(mlp_shap, dummy_model, X_test, y_test)
    shap_values = _calculate_shap(mlp_shap, X_train)
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
        _build_ale_plots(
            mlp_ale,
            X_train,
            config['mc_columns'],
            model_name
        )
    return contribution_df


def _load_train_test(
        train_name: str,
        test_name: str,
        X_columns: list,
        y_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Load train and test set and split it up to sets containing features and target variable only. Additionally
    train is split to train and valid set.
    """
    train = pd.read_parquet('engagement/machine_learning/data/{}.parquet'.format(train_name))
    test = pd.read_parquet('engagement/machine_learning/data/{}.parquet'.format(test_name))
    X_train, X_valid, y_train, y_valid = train_test_split(
        train[X_columns],
        train[y_column],
        test_size=0.2,
        shuffle=True,
        random_state=111
    )
    X_test = test[X_columns]
    y_test = test[y_column]
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def _define_mlp(X_train: pd.DataFrame) -> Sequential:
    """
    Define layers for mlp and compile the model
    """
    n_columns = len(X_train.columns)
    mlp = Sequential()
    # First Layer
    mlp.add(Dense(64, activation='relu', input_dim=n_columns))
    # Dropout Layer
    mlp.add(Dropout(0.5))
    # Second Hidden Layer
    mlp.add(Dense(32, activation='relu'))
    # Output Layer
    mlp.add(Dense(1, activation='sigmoid'))
    # Compile
    mlp.compile(
        optimizer=optimizers.Adam(lr=1e-3),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC(name='prc', curve='PR')]
    )
    return mlp


def _model_for_shap(
        X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame,
        y_valid: pd.Series, model_name: str, load_model: bool = False
) -> Sequential:
    """
    Build and save model which fits as input for SHAP Explainer function
    """
    if load_model:
        mlp_shap = keras.models.load_model('engagement/machine_learning/models/{}.h5'.format(model_name))
    else:
        mlp_shap = _define_mlp(X_train)
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_prc',
            verbose=0,
            patience=20,
            mode='max',
            restore_best_weights=True)
        # Due to the imbalanced distribution of target classes, the batch size is chosen relatively high
        mlp_shap.fit(
            X_train.values, y_train.values, batch_size=2056, epochs=100,
            validation_data=(X_valid.values, y_valid.values),
            verbose=0, callbacks=[early_stopping]
        )
        mlp_shap.save("engagement/machine_learning/models/{}.h5".format(model_name))
    return mlp_shap


def _model_for_ale(
    X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame,
    y_valid: pd.Series
) -> KerasClassifier:
    """
    Build and save model which fits as input for ALE function
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_prc',
        verbose=0,
        patience=20,
        mode='max',
        restore_best_weights=True)
    mlp_ale = KerasClassifier(build_fn=_define_mlp, X_train=X_train, verbose=0, callbacks=[early_stopping])
    mlp_ale.fit(X_train, y_train, batch_size=2056, epochs=100, validation_data=(X_valid, y_valid))
    return mlp_ale


def _model_performance(
        model: Sequential,
        dummy_model: DummyClassifier,
        X_test: pd.DataFrame,
        y_test: pd.Series
) -> None:
    """
    Check the performance (roc_auc, pr_auc) of the model and compare it with random classifier
    """
    mlp_pred = model.predict_proba(X_test)
    dummy_pred = dummy_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, mlp_pred)
    roc_auc_dummy = roc_auc_score(y_test, dummy_pred)
    mlp_precision, mlp_recall, _ = precision_recall_curve(y_test, mlp_pred)
    mlp_pr_auc = auc(mlp_recall, mlp_precision)
    dummy_precision, dummy_recall, _ = precision_recall_curve(y_test, dummy_pred)
    dummy_pr_auc = auc(dummy_recall, dummy_precision)
    print('MLP ROC AUC: %.3f' % roc_auc)
    print('NO SKILL ROC AUC: %.3f' % roc_auc_dummy)
    print('MLP PR AUC: %.3f' % mlp_pr_auc)
    print('NO SKILL PR AUC: %.3f' % dummy_pr_auc)
    return None


def _calculate_shap(model: Sequential, X_train: pd.DataFrame, background_size: int = 1000) -> np.ndarray:
    """
    Calculate SHAP values with predefined background sample. The sample is used for integrating out features
    a background size of 100-1000 is recommended. With background_size > 1000 it gets unreasonably expensive
    """
    shap.initjs()
    background = X_train.sample(n=background_size, random_state=1)
    explainer = shap.DeepExplainer(model, background.values)
    shap_values = explainer.shap_values(X_train.values)
    return shap_values[0]


if __name__ == "__main__":
    """
    Function is executed for engagement features only and for all features
    """
    contribution_engagement = mlp_model(
        config['engagement_features'],
        'mlp_engagement_features',
        compare_engage_non_engage=False,
        load_model=False
    )
    contribution_all_features = mlp_model(
        config['all_features'],
        'mlp_all_features',
        compare_engage_non_engage=True,
        load_model=False
    )
    display(contribution_engagement)
    display(contribution_all_features)

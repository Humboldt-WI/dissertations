import pandas as pd
import numpy as np

from IPython.display import display
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2_contingency


mc_columns = [
    'norm_dwell_time', 'norm_offer_page_seen',
    'norm_paywall_seen', 'norm_registerlogin_page_seen',
    'norm_pv_of_article', 'recency', 'frequency',
    'norm_pi_per_visit', 'norm_checkout_page_seen',
    'bounce_rate'
]

cat_columns = [
    'price', 'page_platform', 'day_of_week', 'day_times',
    'browser', 'genre_top_1', 'genre_top_2', 'location'
]


def correlation_analysis(crosstabs: bool = False) -> None:
    """
    Execute a correlation analysis:
    1. load data
    2. correlation matrix for numerical features with pearson correlation
    3. Variance Inflation Factors for collinear numerical features
    4. correlation matrix for categorical features with cramer's V correlation
    """
    data = _load_data('train_test_prepared.parquet')
    _corr_matrix_numeric(data, mc_columns)
    _vif(data, mc_columns)
    _vif_without_checkout_page(data, mc_columns)
    if crosstabs:
        _print_crosstabs(data, cat_columns)
    _apply_cramers_v(data, cat_columns)


def _load_data(dataname: str) -> pd.DataFrame:
    """
    Load clickstream data with all attributes
    """
    df = pd.read_parquet('engagement/machine_learning/data/{}'.format(dataname))
    return df


def _corr_matrix_numeric(data: pd.DataFrame, features: list) -> None:
    """
    Correlation matrix to analyse correlation between numerical engagement features
    """
    corr = data[features].corr()
    pd.set_option('display.max_columns', 50)
    print('correlation matrix for numerical engagement features:')
    display(corr)


def _vif(data: pd.DataFrame, features: list) -> None:
    """
    Calculate VIF for each feature
    """
    vif_data = pd.DataFrame()
    vif_data["feature"] = data[features].columns
    vif_data["vif"] = [
        variance_inflation_factor(data[features].values, i) for i in range(len(data[features].columns))
    ]
    print('Variance Inflation Factors:')
    display(vif_data)


def _vif_without_checkout_page(data: pd.DataFrame, features: list) -> None:
    """
    Calculate VIF for each feature but 'checkout_page_seen'
    """
    vif_data = pd.DataFrame()
    features.remove('norm_checkout_page_seen')
    vif_data["feature"] = data[features].columns
    vif_data["vif"] = [
        variance_inflation_factor(data[features].values, i) for i in range(len(data[features].columns))
    ]
    print('Variance Inflation Factors:')
    display(vif_data)


def _print_crosstabs(data: pd.DataFrame, features: list) -> None:
    """
    Compute crosstabs to make sure the conditions for the Chi-squared test of independence are met
    """
    cols = data[features].columns.to_list()
    for i in range(len(cols) - 1):
        for j in range(i + 1, len(cols)):
            print(pd.crosstab(data[cols[i]], data[cols[j]]), "\n")


def _cramers_v(var1: pd.Series, var2: pd.Series) -> np.float64:
    """
    Define the function for cramer's v in order to apply it in the next step
    """
    confusion_matrix = pd.crosstab(var1, var2)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))


def _apply_cramers_v(data: pd.DataFrame, features: list) -> None:
    """
    Create matrix with cramer's v values
    """
    rows = []
    for var1 in data[features].columns:
        col = []
        for var2 in data[features].columns:
            col.append(round(_cramers_v(data[var1], data[var2]), 2))
        rows.append(col)
    cramer_matrix = pd.DataFrame(np.array(rows), columns=data[features].columns, index=data[features].columns)
    print('Cramer\'s V matrix:')
    display(cramer_matrix)


if __name__ == "__main__":
    correlation_analysis()

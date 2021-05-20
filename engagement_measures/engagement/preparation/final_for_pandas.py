from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.dataframe import DataFrame

from utils import _create_spark_session, _read_file


def final_for_pandas(spark: SparkSession) -> tuple[DataFrame, DataFrame]:
    """
    Prepare the dataset so that it is suitable for local processing as pandas df
    """
    sessions = _read_file(spark, 'sessions_features.parquet')
    sessions = _filter_after_august(sessions)
    sessions = _filter_price_displayed(sessions)
    sessions = _filter_after_fifth_session(sessions)
    sessions = _filter_sessions_before_conv(sessions)
    train, test = _train_test_split(sessions)
    train, test = _normalizing(train, test)
    train, test = _columns_for_pandas(train, test)
    return train, test


def _filter_after_august(df: DataFrame) -> DataFrame:
    """
    Keep only the sessions after august because the created features in july
    are not built on enough browsing history (30 days are needed)
    """
    df = (
        df
        .filter(
            F.month(
                F.col("time_stamp_start")
            )
            > 8
        )
    )
    return df


def _filter_price_displayed(df: DataFrame) -> DataFrame:
    """
    Keep only the sessions where a price was displayed. That will be the
    basis to predict whether a subscription was taken out.
    """
    df = (
        df
        .filter(
            F.col('price')
            .isNotNull()
        )
    )
    return df


def _filter_after_fifth_session(df: DataFrame) -> DataFrame:
    """
    As it was defined that a user needs at least 6 visits to get counted and the
    conversion session will always be the last in the user's record, it is impossible
    that a user converts within the first 5 session. That is the reason why they get
    dropped here.
    """
    df = (
        df
        .filter(
            F.col('session_number') > 5
        )
    )
    return df


def _filter_sessions_before_conv(df: DataFrame) -> DataFrame:
    """
    Users who convert only stay with the session in which they convert. This way the ML
    models do not learn from previous sessions of converting users and the results become
    more robust.
    """
    window = (
        Window
        .partitionBy('cookie_id')
        .orderBy(F.desc('time_stamp_start'))
        .rangeBetween(Window.unboundedPreceding, 0)
    )
    df = (
        df
        .withColumn(
            'before_order',
            F.sum('orders').over(window)
        )
        .withColumn(
            'before_order',
            F.sum('before_order').over(window)
        )
        .filter(
            F.col('before_order') < 2).drop('before_order')
    )
    return df


def _train_test_split(df: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Split the df in train and test set
    """
    train, test = (
        df
        .randomSplit(
            [0.7, 0.3],
            seed=123
        )
    )
    return train, test


def _normalizing(train: DataFrame, test: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Normalize the train set using cumulative distribution function and join with
    the test set afterwards. This will prevent data leakage.
    """
    train = (
        train
        .withColumn(
            'norm_pis',
            F.cume_dist().over(
                Window.orderBy('sum_pis')
            )
        )
        .withColumn(
            'norm_dwell_time',
            F.cume_dist().over(
                Window.orderBy('sum_dwell_time')
            )
        )
        .withColumn(
            'norm_offer_page_seen',
            F.cume_dist().over(
                Window.orderBy(
                    'sum_offer_page_seen')
            )
        )
        .withColumn(
            'norm_paywall_seen',
            F.cume_dist().over(
                Window.orderBy('sum_paywall_seen')
            )
        )
        .withColumn(
            'norm_registerlogin_page_seen',
            F.cume_dist().over(
                Window.orderBy('sum_registerlogin_page_seen')
            )
        )
        .withColumn(
            'norm_checkout_page_seen',
            F.cume_dist().over(
                Window.orderBy('sum_checkout_page_seen')
            )
        )
        .withColumn(
            'norm_visits',
            F.cume_dist().over(
                Window.orderBy('sum_visits')
            )
        )
        .withColumn(
            'norm_pi_per_visit',
            F.cume_dist().over(
                Window.orderBy('pis_per_visit')
            )
        )
        .withColumn(
            'norm_pv_of_article',
            F.cume_dist().over(
                Window.orderBy('sum_pv_of_article')
            )
        )
    )
    pis = (
        train
        .select('norm_pis', 'sum_pis')
        .dropDuplicates()
    )
    duration = (
        train
        .select('norm_dwell_time', 'sum_dwell_time')
        .dropDuplicates()
    )
    offer_page_seen = (
        train
        .select('norm_offer_page_seen', 'sum_offer_page_seen')
        .dropDuplicates()
    )
    paywall_seen = (
        train
        .select('norm_paywall_seen', 'sum_paywall_seen')
        .dropDuplicates()
    )
    registerlogin_page_seen = (
        train
        .select('norm_registerlogin_page_seen', 'sum_registerlogin_page_seen')
        .dropDuplicates()
    )
    checkout_page_seen = (
        train.select('norm_checkout_page_seen', 'sum_checkout_page_seen')
        .dropDuplicates()
    )
    visits = (
        train.select('norm_visits', 'sum_visits')
        .dropDuplicates()
    )
    pi_per_visit = (
        train
        .select('norm_pi_per_visit', 'pis_per_visit')
        .dropDuplicates()
    )
    pv_of_article = (
        train
        .select('norm_pv_of_article', 'sum_pv_of_article')
        .dropDuplicates()
    )
    test = (
        test
        .join(
            pis,
            'sum_pis',
            how='left'
        )
        .join(
            duration,
            'sum_dwell_time',
            how='left'
        )
        .join(
            offer_page_seen,
            'sum_offer_page_seen',
            how='left'
        )
        .join(
            paywall_seen,
            'sum_paywall_seen',
            how='left'
        )
        .join(
            registerlogin_page_seen,
            'sum_registerlogin_page_seen',
            how='left'
        )
        .join(
            checkout_page_seen,
            'sum_checkout_page_seen',
            how='left'
        )
        .join(
            visits,
            "sum_visits",
            how="left"
        )
        .join(
            pi_per_visit,
            'pis_per_visit',
            how='left'
        )
        .join(
            pv_of_article,
            'sum_pv_of_article',
            how='left'
        )
    )
    test = (
        test
        .dropna(
            subset=(
                'norm_pis', 'norm_dwell_time', 'norm_offer_page_seen', 'norm_paywall_seen',
                'norm_registerlogin_page_seen', 'norm_checkout_page_seen', 'norm_visits',
                'norm_pi_per_visit', 'norm_pv_of_article'
            )
        )
    )
    return train, test


def _columns_for_pandas(train: DataFrame, test: DataFrame) -> tuple[DataFrame, DataFrame]:
    """
    Choose only those columns which are used in the pandas transformations
    """
    columns = [
        'orders', 'page_platform', 'browser', 'price', 'user_login', 'genre_top_1',
        'genre_top_2', 'location', 'day_of_week', 'day_times', 'bounce_rate',
        'clumpiness_score', 'frequency', 'recency', 'norm_dwell_time', 'norm_offer_page_seen',
        'norm_pi_per_visit', 'norm_paywall_seen', 'norm_registerlogin_page_seen',
        'norm_checkout_page_seen', 'norm_pv_of_article', 'cookie_id'
    ]
    train = train.select(*columns)
    test = test.select(*columns)
    return train, test


if __name__ == "__main__":
    with _create_spark_session() as spark:
        train, test = final_for_pandas(spark)
        test.show()
        # train.repartition(1).write.parquet('engagement/machine_learning/data/train.parquet')
        # test.repartition(1).write.parquet('engagement/machine_learning/data/test.parquet')

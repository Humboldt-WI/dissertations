import random
import operator

from itertools import chain
from math import log10
from collections import Counter

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import StringType, IntegerType, MapType, ArrayType, StructType, StructField
from pyspark.sql.dataframe import DataFrame
import pandas as pd

from utils import _create_spark_session, _read_file


def feature_engineering(spark: SparkSession) -> DataFrame:
    """
    Create the features which can be used as input for ML models
    """
    session_df = _read_file(spark, 'sessions.parquet')
    session_df = _create_date_of_action(session_df)
    day_df = _day_aggregation(session_df)
    day_df = _date_in_seconds(day_df)
    day_df = _thirty_days_browsing(day_df)
    day_df = _frequency(day_df)
    day_df = _recency(day_df)
    day_df = _clumpiness(day_df)
    session_df = _join_session_day_level(session_df, day_df)
    session_df = _genre_list(session_df)
    session_df = _genre_ordered(session_df)
    session_df = _genre_preference(session_df)
    session_df = _extract_geo_info(session_df)
    session_df = _day_of_week(session_df)
    session_df = _day_times(session_df)
    session_df = _clean_browser_type(session_df)
    session_df = _clean_user_login(session_df)
    return session_df


def _create_date_of_action(df: DataFrame) -> DataFrame:
    """
    Create a new column which includes the date a user was tracked
    """
    df = df.withColumn(
        'date_of_action',
        F.to_date(
            F.col('time_stamp_start')
        )
    )
    return df


def _day_aggregation(df: DataFrame) -> DataFrame:
    """
    Aggregate for every user on daily level
    """
    df = (
        df
        .groupBy('cookie_id', 'date_of_action')
        .agg(
            F.countDistinct('session_id').alias('visits'),
            F.sum('dwell_time').alias('dwell_time'),
            F.sum('pis').alias('pis'),
            F.sum('paywall_seen').alias('paywall_seen'),
            F.sum('offer_page_seen').alias('offer_page_seen'),
            F.sum('registerlogin_page_seen').alias('registerlogin_page_seen'),
            F.sum('checkout_page_seen').alias('checkout_page_seen'),
            F.sum('pv_of_article').alias('pv_of_article'),
            F.sum('bounce').alias('bounce'),
            F.max('user_login_status').alias('user_login_status')
        )
    )
    return df


def _date_in_seconds(df: DataFrame) -> DataFrame:
    """
    To create a sliding window containing 30 days of the user's browsing behavior,
    the date in seconds is needed
    """
    df = df.withColumn(
        'date_in_seconds',
        F.col('date_of_action').astype('Timestamp').cast('long')
    )
    return df


def _thirty_days_browsing(df: DataFrame) -> DataFrame:
    """
    Create window, partition by cookie id, order by date a user was activ
    and define a range between the previous 30 days (2.592e+6 seconds) and the day before
    current day (-1)
    (-1 second) -> this is done to avoid data leakage from current session
    """
    window = (
        Window
        .partitionBy('cookie_id')
        .orderBy('date_in_seconds')
        .rangeBetween(-2592000, -1)
    )
    df = (
        df
        .withColumn('sum_pis', F.sum(F.col('pis')).over(window))
        .withColumn('sum_dwell_time', F.sum(F.col('dwell_time')).over(window))
        .withColumn('sum_visits', F.sum(F.col('visits')).over(window))
        .withColumn('sum_paywall_seen', F.sum(F.col('paywall_seen')).over(window))
        .withColumn('sum_offer_page_seen', F.sum(F.col('offer_page_seen')).over(window))
        .withColumn('sum_registerlogin_page_seen', F.sum(F.col('registerlogin_page_seen')).over(window))
        .withColumn('sum_checkout_page_seen', F.sum(F.col('checkout_page_seen')).over(window))
        .withColumn('sum_pv_of_article', F.sum(F.col('pv_of_article')).over(window))
        .withColumn('sum_bounce', F.sum(F.col('bounce')).over(window))
        .withColumn('user_login', F.max(F.col('user_login_status')).over(window))
        .withColumn('bounce_rate', F.col('sum_bounce') / F.col('sum_visits'))
        .withColumn('pis_per_visit', F.col('sum_pis') / F.col('sum_visits'))
    )
    columns_to_drop = [
        'pis', 'offer_page_seen', 'paywall_seen', 'registerlogin_page_seen', 'checkout_page_seen',
        'dwell_time', 'pv_of_article', 'user_login_status', 'bounce', 'sum_bounce'
    ]
    df = df.drop(*columns_to_drop)
    return df


def _frequency(df: DataFrame) -> DataFrame:
    """
    calculate on how many days a user was active in the last thirty days (including current one) and normalize it
    by dividing the number of days by the total possible days (30)
    """
    window = (
        Window
        .partitionBy('cookie_id')
        .orderBy('date_in_seconds')
        .rangeBetween(-2592000, 0)
    )
    df = df.withColumn(
        'frequency',
        F.count(
            F.col('cookie_id')
        ).over(window) / 30
    )
    return df


def _recency(df: DataFrame) -> DataFrame:
    """
    Calculate how many days have been past between the current date of action and the
    last time the user has visited the page
    For normalizing the number of days is divided by 31 so differences higher than
    30 days get >=1 which is easier to filter since time of interes are only 30
    days
    """
    window = (
        Window
        .partitionBy('cookie_id')
        .orderBy('date_of_action')
    )
    df = df.withColumn(
        'recency',
        F.datediff(
            F.col('date_of_action'),
            F.lag(
                F.col('date_of_action'),
                1
            )
            .over(window)
        ) / 31
    )
    df = df.withColumn(
        'recency',
        F.when(
            F.col('recency') > 1, 0
        )
        .otherwise(
            F.when(
                F.col('recency').isNull(), 0
            )
            .otherwise(
                1 - F.col('recency')
            )
        )
    )
    return df


def _mont_carlo_simulation(N: int, M: int) -> dict:
    """
    In order to calculate the z_scores for clumpiness a monte carlo simulation
    is performed
    """
    clumpiness_dict = {}
    for n in range(1, N + 1):
        clumpiness = []
        for m in range(1, M + 1):
            sample = random.sample(range(1, N + 1), n)
            sample.extend([0, N + 1])
            sample.sort()
            dif = [j - i for i, j in zip(sample[:-1], sample[1:])]
            sum = 0
            for i in range(0, len(dif)):
                c = (dif[i] / (N + 1)) * log10((dif[i] / (N + 1)))
                sum = sum + c
            clumpiness.append(1 + sum / log10(n + 1))
        clumpiness_dict[str(n)] = clumpiness
    pandas_clumpiness = pd.DataFrame.from_dict(clumpiness_dict)

    z_scores = {}
    for i in pandas_clumpiness.columns.tolist():
        z_scores[i] = pandas_clumpiness[i].quantile([.95]).values[0]
    return z_scores


def _clumpiness(df: DataFrame) -> DataFrame:
    """
    The following calculations are based on results from the paper "Predicting Customer Value
    Using Clumpiness: From RFM to RFMC" by Zhang et al. (2015).
    In the present case, an observation period of 30 days is assumed (N=30)
    """
    window1 = (
        Window
        .partitionBy('cookie_id')
        .orderBy('date_in_seconds')
        .rangeBetween(-2592000, -1)
    )
    window2 = (
        Window
        .partitionBy('cookie_id')
        .orderBy('date_of_action')
    )
    # compute how often a user has been visiting the website in the last 30
    # days (active_days).
    df = df.withColumn(
        "active_days",
        F.count(
            F.col('cookie_id')
        )
        .over(window1)
    )
    # calculate the difference between each visit, starting with the current one
    for i in range(1, 31):
        df = df.withColumn(
            'diff_{0}'.format(i),
            F.datediff(
                F.lag(
                    F.col('date_of_action'), i - 1
                )
                .over(window2),
                F.lag(
                    F.col('date_of_action'), i
                )
                .over(window2)
            )
        )
    # the formula for clumpiness also requires calculating the difference between the last
    # visited day within the 30 days and N + 1 (31)
    df = (
        df
        .withColumn(
            'last_date',
            F.first(
                F.col('date_of_action')
            )
            .over(window1)
        )
        .withColumn(
            'diff_0',
            31 - (
                F.datediff(
                    F.col('date_of_action'),
                    F.col('last_date')
                )
            )
        )
    )
    # Next, the numerator of the clumpiness formula is calculated, beginning with the following part:
    # log(xi/N+1)*(xi/N+1)
    for i in range(0, 31):
        df = df.withColumn(
            'diff_{0}'.format(i),
            (
                    F.col('diff_{0}'.format(i)) / 31
            ) * (
                F.log10(
                    F.col(
                        'diff_{0}'.format(i)
                    ) / 31
                )
            )
        )
    # sum the results from previous step, depending on the active days
    df = (
        df
        .withColumn(
            'clumpiness_score',
            F.lit(None)
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 1,
                F.col('diff_0') + F.col('diff_1')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 2,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 3,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 4,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 5,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 6,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 7,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 8,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 9,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 10,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 11,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 12,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 13,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 14,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 15,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 16,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 17,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 18,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 19,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18') + F.col('diff_19')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 20,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18') + F.col('diff_19') + F.col('diff_20')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 21,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18') + F.col('diff_19') + F.col('diff_20') +
                F.col('diff_21')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 22,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18') + F.col('diff_19') + F.col('diff_20') +
                F.col('diff_21') + F.col('diff_22')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 23,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18') + F.col('diff_19') + F.col('diff_20') +
                F.col('diff_21') + F.col('diff_22') + F.col('diff_23')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 24,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18') + F.col('diff_19') + F.col('diff_20') +
                F.col('diff_21') + F.col('diff_22') + F.col('diff_23') +
                F.col('diff_24')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 25,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18') + F.col('diff_19') + F.col('diff_20') +
                F.col('diff_21') + F.col('diff_22') + F.col('diff_23') +
                F.col('diff_24') + F.col('diff_25')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 26,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18') + F.col('diff_19') + F.col('diff_20') +
                F.col('diff_21') + F.col('diff_22') + F.col('diff_23') +
                F.col('diff_24') + F.col('diff_25') + F.col('diff_26')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 27,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18') + F.col('diff_19') + F.col('diff_20') +
                F.col('diff_21') + F.col('diff_22') + F.col('diff_23') +
                F.col('diff_24') + F.col('diff_25') + F.col('diff_26') +
                F.col('diff_27')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 28,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18') + F.col('diff_19') + F.col('diff_20') +
                F.col('diff_21') + F.col('diff_22') + F.col('diff_23') +
                F.col('diff_24') + F.col('diff_25') + F.col('diff_26') +
                F.col('diff_27') + F.col('diff_28')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 29,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18') + F.col('diff_19') + F.col('diff_20') +
                F.col('diff_21') + F.col('diff_22') + F.col('diff_23') +
                F.col('diff_24') + F.col('diff_25') + F.col('diff_26') +
                F.col('diff_27') + F.col('diff_28') + F.col('diff_29')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
        .withColumn(
            'clumpiness_score',
            F.when(
                F.col('active_days') == 30,
                F.col('diff_0') + F.col('diff_1') + F.col('diff_2') +
                F.col('diff_3') + F.col('diff_4') + F.col('diff_5') +
                F.col('diff_6') + F.col('diff_7') + F.col('diff_8') +
                F.col('diff_9') + F.col('diff_10') + F.col('diff_11') +
                F.col('diff_12') + F.col('diff_13') + F.col('diff_14') +
                F.col('diff_15') + F.col('diff_16') + F.col('diff_17') +
                F.col('diff_18') + F.col('diff_19') + F.col('diff_20') +
                F.col('diff_21') + F.col('diff_22') + F.col('diff_23') +
                F.col('diff_24') + F.col('diff_25') + F.col('diff_26') +
                F.col('diff_27') + F.col('diff_28') + F.col('diff_29') +
                F.col('diff_30')
            )
            .otherwise(
                F.col('clumpiness_score')
            )
        )
    )
    # drop columns which are not needed anymore
    for i in range(0, 31):
        df = df.drop(
            F.col('diff_{0}'.format(i))
        )
    df = df.drop(F.col('last_date'))
    # the numerator is substituted into the rest of the formula
    df = df.withColumn(
        'clumpiness_score',
        1+F.col('clumpiness_score') /
        F.log10(
            F.col('active_days')+1
        )
    )
    # In order to check the significance of the calculated clumpiness score a statistical test is needed.
    # The null hypothesis is random sampling without replacement, where n (the number of events) and
    # N (the number of trials) are known, and Monte Carlo simulation is applied to compute the Z-table,
    # the table of clumpiness critical values.
    z_table = _mont_carlo_simulation(30, 5000)
    # in order to map the z_scores to the number of active days the 'active_days' column is converted to
    # string type
    df = df.withColumn(
        'active_days',
        F.col('active_days').cast(StringType())
    )
    mapping_expr = F.create_map(
        [F.lit(x) for x in chain(*z_table.items())]
    )
    df = df.withColumn(
        'z_score_clumpiness',
        mapping_expr.getItem(F.col('active_days'))
    )
    return df


def _join_session_day_level(session: DataFrame, day: DataFrame) -> DataFrame:
    """
    Join information created on daily basis with session level
    """
    join = session.join(
        day,
        ['cookie_id', 'date_of_action'],
        how='inner'
    )
    return join


def _genre_list(df: DataFrame) -> DataFrame:
    """
    Create column containing all genres a user have seen in the last 30 days
    (excluding current day)
    """
    window = (
        Window
        .partitionBy('cookie_id')
        .orderBy('date_in_seconds')
        .rangeBetween(-2592000, -1)
    )
    df = df.withColumn(
        'genre_list',
        F.flatten(
            F.collect_list(
                F.col('page_section_1')
            )
            .over(window)
        )
    )
    return df


def sort_dict_f(x):
    """
    Helper function for udf to sort items in dict
    """
    sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_x


def _genre_ordered(df: DataFrame) -> DataFrame:
    """
    Sort genre by number of occurrences
    """
    udf_counter = F.udf(
        lambda x: dict(Counter(x)),
        MapType(
            StringType(),
            IntegerType()
        )
    )
    df = df.withColumn(
        'genre_ordered',
        udf_counter('genre_list')
    )
    schema = ArrayType(
        StructType(
            [
                StructField(
                    'genre',
                    StringType(),
                    False
                ),
                StructField(
                    'count',
                    IntegerType(),
                    False
                )
            ]
        )
    )
    udf_sorter = F.udf(sort_dict_f, schema)
    df = (
        df
        .withColumn(
            'genre_ordered',
            udf_sorter('genre_ordered')
        )
        .withColumn(
            'genre_preference',
            F.col('genre_ordered').getItem('genre')
        )
    )
    return df


def _genre_preference(df: DataFrame) -> DataFrame:
    """
    Pick top 1 (most viewed genre) and top 2 ( second most viewed genre)
    for every user. A genre is only counted as a top genre if the
    user has seen it at least 3 times.
    """
    df = (
        df
        .withColumn(
            'genre_top_1',
            df.genre_preference[0]
        )
        .withColumn(
            'genre_top_2',
            df.genre_preference[1]
        )
        .withColumn(
            'count_array',
            F.col('genre_ordered').getItem('count')
        )
        .withColumn(
            'genre_top_1',
            F.when(
                F.col('count_array')[0] > 2,
                F.col('genre_top_1')
            )
            .otherwise(None)
        )
        .withColumn(
            'genre_top_2',
            F.when(
                F.col('count_array')[1] > 2,
                F.col('genre_top_2')
            )
            .otherwise(None)
        )
        .drop(
            'genre_list', 'count_array',
            'genre_preference', 'genre_ordered',
            'page_section_1'
        )
    )
    return df


def _extract_geo_info(df: DataFrame) -> DataFrame:
    """
    1. extracting the states for germany
    2. extracting german speaking countries
    3. all other countries are reffered to 'other_country'
    4. drop rows where location was tracked as germany but no related state is given
    """
    df = (
        df
        .withColumn(
            'location',
            F.when(
                F.col('geo_state').like('%germany%'),
                F.regexp_replace('geo_state', r'\(.*\)', '')
            )
            .otherwise(F.col('geo_state'))
        )
        .withColumn(
            'location',
            F.when(
                F.col('location').like('%switzerland%'),
                'switzerland'
            )
            .otherwise(F.col('location'))
        )
        .withColumn(
            'location',
            F.when(
                F.col('location').like('%austria%'),
                'austria'
            )
            .otherwise(F.col('location'))
        )
        .withColumn(
            'location',
            F.when(
                F.col('location').like('%luxembourg%'),
                'luxembourg'
            )
            .otherwise(F.col('location'))
        )
        .withColumn(
            'location',
            F.when(
                F.col('location').rlike(r'\(.*\)'),
                'other_country'
            )
            .otherwise(F.col('location'))
        )
        .filter(
            ~F.col('location').like('%germany%')
            )
        .drop('geo_state')
    )
    return df


def _day_of_week(df: DataFrame) -> DataFrame:
    """
    Extract day of week from timestamp
    """
    df = df.withColumn(
        'day_of_week',
        F.date_format('date_of_action', 'EEEE')
    )
    return df


def _day_times(df: DataFrame) -> DataFrame:
    """
    Define time of day (morning, noon, evening, night) based on hours
    """
    df = (
        df
        .withColumn(
            'hour',
            F.hour(F.col('time_stamp_start'))
        )
        .withColumn(
            'day_times',
            F.when(
                (F.col('hour') >= 6) &
                (F.col('hour') < 11),
                'Morning'
            )
            .otherwise(None)
        )
        .withColumn(
            'day_times',
            F.when(
                (F.col('hour') >= 11) &
                (F.col('hour') < 18),
                'Noon'
            )
            .otherwise(F.col('day_times'))
        )
        .withColumn(
            'day_times',
            F.when(
                (F.col('hour') >= 18) &
                (F.col('hour') <= 23),
                'Evening'
            )
            .otherwise(F.col('day_times'))
        )
        .withColumn(
            'day_times',
            F.when(
                (F.col('hour') >= 0) &
                (F.col('hour') < 6),
                'Night'
            )
            .otherwise(F.col('day_times'))
        )
        .drop('hour')
    )
    return df


def _clean_browser_type(df: DataFrame) -> DataFrame:
    """
    Filter only the most used browser types, all other browsers are summed
    to 'other_browser'
    """
    browsers = [
        'amazon', 'apple', 'google', 'microsoft',
        'mozilla', 'opera', 'samsung',
    ]
    df = (
        df
        .withColumn(
            'browser',
            F.when(
                F.col('browser_type').isin(browsers),
                F.col('browser_type')
            )
            .otherwise('other_browser')
        )
    )
    return df


def _clean_user_login(df: DataFrame) -> DataFrame:
    """
    If a user logged in multiple times in the last 30 days it is just counted
    as one
    """
    df = df.withColumn(
        "user_login",
        F.when(
            F.col("user_login"),
            1
        )
        .otherwise(0)
    )
    return df


if __name__ == "__main__":
    with _create_spark_session() as spark:
        session = feature_engineering(spark)
        session.repartition(1).write.parquet('engagement/preparation/data/sessions_features.parquet')

from itertools import chain

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.dataframe import DataFrame

from config import config
from utils import _create_spark_session, _read_file


def session_preparation(spark: SparkSession) -> DataFrame:
    """
    Build session logic in order to not loose important information and be able to engineer
    features
    """
    event_df = _read_file(spark, 'events.parquet')
    event_df = _cleaning(event_df)
    session_df = _session_aggregation(event_df)
    session_df = _create_bounce(session_df)
    session_df = _logic_paywall_seen(session_df)
    prices = config['prices']
    session_df = _add_price(session_df, prices)
    session_df = _drop_tracking_issues(session_df)
    session_df = _clean_conv(session_df)
    session_df = _remove_sessions_after_conv(session_df)
    session_df = _remove_users_with_conv_before_time_range(session_df)
    session_df = _at_least_five_visits(session_df)
    session_df = _order_session_number(session_df)
    session_df = _remove_users_with_no_offer(session_df)
    return session_df


def _cleaning(df: DataFrame) -> DataFrame:
    """
    Bring 'time_stamp' and 'products' into right format
    """
    df = (
        df
        .withColumn('time_stamp', F.to_timestamp(F.col('time_stamp')))
        .withColumn('products', F.col('products').cast('string'))
    )
    return df


def _session_aggregation(df: DataFrame) -> DataFrame:
    """
    Aggregate events on session level
    """
    session_df = df.groupBy('session_id').agg(
        F.first(F.col('cookie_id'), True).alias('cookie_id'),
        F.min(F.col('time_stamp')).alias('time_stamp_start'),
        F.max(F.col('time_stamp')).alias('time_stamp_end'),
        F.sum(F.col('page_views')).alias('pis'),
        F.sum(F.col('pv_of_article_event')).alias('pv_of_article'),
        F.sum(F.col('total_seconds_spent')).alias('dwell_time'),
        F.sum(F.col('paywall_event')).alias("paywall_seen"),
        F.sum(F.col('offer_page_event')).alias('offer_page_seen'),
        F.sum(F.col('registerlogin_page_event')).alias('registerlogin_page_seen'),
        F.sum(F.col('checkout_page_event')).alias('checkout_page_seen'),
        F.sum(F.col('clicks_to_page')).alias('bounce'),
        F.max(F.col('user_login_status')).alias('user_login_status'),
        F.max(F.col('user_subscription_status')).alias('user_subscription_status'),
        F.max(F.col('page_premium_status')).alias('page_premium_status'),
        F.collect_set(F.col('page_section')).alias('page_section_1'),
        F.first(F.col('page_platform')).alias('page_platform'),
        F.first(F.col('browser_type'), True).alias('browser_type'),
        F.first(F.col('geosegmentation_states'), True).alias('geo_state'),
        F.collect_set(F.col('products')).alias('products'),
        F.sum(F.col('orders')).alias('orders')
    )
    return session_df


def _create_bounce(df: DataFrame) -> DataFrame:
    """
    Edit bounce column: when a user had only one click in the whole session it is counted as a bounce
    """
    df = df.withColumn('bounce', F.when(F.col('bounce') == 1, 1).otherwise(0))
    return df


def _logic_paywall_seen(df: DataFrame) -> DataFrame:
    """
    Tracking paywall hits seems to have problems.
    New logic: when a premium article was clicked and the user suscription status is 'False', it is
    counted as a paywall hit
    """
    df = df.withColumn(
        'paywall_seen',
        F.when(
            (
                (F.col('page_premium_status') == True) &
                (F.col('user_subscription_status') == False) &
                (F.col('paywall_seen') == 0)
            ),
            F.col('paywall_seen') + 1
        )
        .otherwise(
            F.col('paywall_seen')
        )
    )
    return df


def _add_price(df: DataFrame, prices: dict) -> DataFrame:
    """
    Use product ids to map prices. Sometimes more than one product id is given within a session. In that case the
    the lowest price is chosen and the others are dropped.
    """
    df = (
        df
        .withColumn('product_1', F.col('products')[0])
        .withColumn('product_2', F.col('products')[1])
        .withColumn('product_3', F.col('products')[2])
    )
    mapping_expr = F.create_map([F.lit(x) for x in chain(*prices.items())])
    df = (
        df
        .withColumn('product_1', mapping_expr.getItem(F.col('product_1')))
        .withColumn('product_2', mapping_expr.getItem(F.col('product_2')))
        .withColumn('product_3', mapping_expr.getItem(F.col('product_3')))
    )
    df = (
        df
        .withColumn(
            'price',
            F.when(
                (
                    (F.col('product_1') == 'price_7') |
                    (F.col('product_2') == 'price_7') |
                    (F.col('product_3') == 'price_7')
                ),
                'price_7'
            )
            .otherwise(None)
        )
        .withColumn(
            'price',
            F.when(
                (
                    (F.col('product_1') == 'price_6') |
                    (F.col('product_2') == 'price_6') |
                    (F.col('product_3') == 'price_6')
                ),
                'price_6'
            )
            .otherwise(F.col('price'))
        )
        .withColumn(
            'price',
            F.when(
                (
                    (F.col('product_1') == 'price_5') |
                    (F.col('product_2') == 'price_5') |
                    (F.col('product_3') == 'price_5')
                ),
                'price_5'
            )
            .otherwise(F.col('price'))
        )
        .withColumn(
            'price',
            F.when(
                (
                    (F.col('product_1') == 'price_4') |
                    (F.col('product_2') == 'price_4') |
                    (F.col('product_3') == 'price_4')
                ),
                'price_4'
            )
            .otherwise(F.col('price'))
        )
        .withColumn(
            'price',
            F.when(
                (
                    (F.col('product_1') == 'price_3') |
                    (F.col('product_2') == 'price_3') |
                    (F.col('product_3') == 'price_3')
                ),
                'price_3'
            )
            .otherwise(F.col('price'))
        )
        .withColumn(
            'price',
            F.when(
                (
                    (F.col('product_1') == 'price_2') |
                    (F.col('product_2') == 'price_2') |
                    (F.col('product_3') == 'price_2')
                ),
                'price_2'
            )
            .otherwise(F.col('price'))
        )
        .withColumn(
            'price',
            F.when(
                (
                    (F.col('product_1') == 'price_1') |
                    (F.col('product_2') == 'price_1') |
                    (F.col('product_3') == 'price_1')
                ),
                'price_1'
            )
            .otherwise(F.col('price'))
        )
    )
    df = df.drop('products', 'product_1', 'product_2', 'product_3')
    return df


def _drop_tracking_issues(df: DataFrame) -> DataFrame:
    """
    Filter sessions which seem to have tracking issues
    """
    df = df.filter(
        (F.col('pis') > 0) &
        (F.col('dwell_time') > 0)
    )
    return df


def _clean_conv(df: DataFrame) -> DataFrame:
    """
    It happens that in one session more than one order takes place. Those cases get counted as one order.
    """
    df = df.withColumn(
        "orders",
        F.when(
            F.col("orders") > 1, 1
        )
        .otherwise(
            F.col("orders")
        )
    )
    return df


def _remove_sessions_after_conv(df: DataFrame) -> DataFrame:
    """
    In order to identify when the first conversion took place (since one user can have multiple transactions),
    the calculated conversions are summed two times.
    All sessions after the first conversion get dropped.
    """
    window = (
        Window
        .partitionBy('cookie_id')
        .orderBy('time_stamp_start')
        .rangeBetween(Window.unboundedPreceding, 0)
    )
    df = (
        df
        .withColumn('subscription_day', F.sum('orders').over(window))
        .withColumn('subscription_day', F.sum('subscription_day').over(window))
    )
    df = df.filter(df.subscription_day < 2)
    return df


def _remove_users_with_conv_before_time_range(df: DataFrame) -> DataFrame:
    """
    Remove sessions from users who subscribed before time range of interest
    """
    window = Window.partitionBy('cookie_id')
    df = (
        df
        .withColumn(
            'sub_before_order',
            F.when(
                (
                    (F.col('user_subscription_status')) &
                    (F.col('orders') == 0)
                ),
                True
            )
            .otherwise(False)
        )
        .withColumn(
            'sub_before_order',
            F.max(F.col('sub_before_order')).over(window)
        )
    )
    df = df.filter(df.sub_before_order == False).drop(F.col('sub_before_order'))
    return df


def _at_least_five_visits(df: DataFrame) -> DataFrame:
    """
    Filter only users who are known for at least 6 sessions
    """
    window = Window.partitionBy('cookie_id')
    df = (
        df
        .withColumn('total_visits', F.count(F.col('session_id')).over(window))
        .filter(F.col('total_visits') > 5).drop('total_visits')
    )
    return df


def _order_session_number(df: DataFrame) -> DataFrame:
    """
    Create a column containing the order within the sessions of a user.
    This will be used to later identify the sessions that happened before
    a user's sixth session.
    """
    window = Window.partitionBy("cookie_id").orderBy("time_stamp_start")
    df = df.withColumn("session_number", F.row_number().over(window))
    return df


def _remove_users_with_no_offer(df: DataFrame) -> DataFrame:
    """
    Keep only the users who saw at least one of the defined prices in their surf history and who were
    active after august since at least 30 days of history from a user is needed for the features
    and the dataset starts in july
    """
    df1 = df
    df2 = (
        df
        .filter(F.col('price').isNotNull())
        .filter(F.month(F.col("time_stamp_start")) > 8)
        .select(F.col('cookie_id')).distinct()
    )
    join = df1.join(df2, 'cookie_id', how='inner')
    return join


if __name__ == "__main__":
    with _create_spark_session() as spark:
        sessions = session_preparation(spark)
        sessions.coalesce(1).write.parquet('engagement/preparation/data/sessions.parquet')



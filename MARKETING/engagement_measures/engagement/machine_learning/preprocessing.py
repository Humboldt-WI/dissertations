import pandas as pd


def concat_prepared_train_test() -> None:
    """
    For further analysis (e.g., correlation) the whole dataset is written to parquet and for the model training the train
    and test set is written to parquet separately
    """
    train = _prepare_clickstream_data('train')
    test = _prepare_clickstream_data('test')
    train_test_concat = pd.concat([train, test], axis=0)
    _descriptive_overview(train_test_concat)
    columns_to_drop = [
        'page_platform', 'price', 'day_of_week', 'day_times', 'browser', 'location',
        'genre_top_1', 'genre_top_2', 'mobile', 'no_top_1', 'no_top_2'
    ]
    train = train.drop(columns=columns_to_drop)
    test = test.drop(columns=columns_to_drop)
    train.to_parquet('engagement/machine_learning/data/train_prepared.parquet', index=False)
    test.to_parquet('engagement/machine_learning/data/test_prepared.parquet', index=False)
    train_test_concat.to_parquet('engagement/machine_learning/data/train_test_prepared.parquet', index=False)


def _prepare_clickstream_data(filename: str) -> pd.DataFrame:
    """
    Prepare the clickstream data in such a way that the ML models can be trained
    """
    clickstream_data = _import_clickstream_data(filename)
    clickstream_data = _fill_na(clickstream_data, ['clumpiness_score', 'bounce_rate'], 0)
    clickstream_data = _fill_na(clickstream_data, ['genre_top_1'], 'no_top_1')
    clickstream_data = _fill_na(clickstream_data, ['genre_top_2'], 'no_top_2')
    clickstream_data = _get_dummies_no_genre(clickstream_data,
                                             ['page_platform', 'price', 'day_of_week', 'day_times', 'browser',
                                              'location'])
    clickstream_data = _get_dummies_genre(clickstream_data, ['genre_top_1', 'genre_top_2'])
    return clickstream_data


def _import_clickstream_data(filename: str) -> pd.DataFrame:
    """
    Import data and print some general information
    """
    data = pd.read_parquet('engagement/machine_learning/data/{}.parquet'.format(filename))
    return data


def _fill_na(df: pd.DataFrame, columns_with_na: list, fill_element: str or int) -> pd.DataFrame:
    """
    Fill cells containing 'na' with a reasonable value
    """
    for column in columns_with_na:
        df[column] = df[column].fillna(fill_element)
    return df


def _get_dummies_no_genre(df: pd.DataFrame, catcolumns: list) -> pd.DataFrame:
    """
    In order to feed the model with categorical information the corresponding columns must be transformed to dummy
    variables. Since there are some differences between the columns how to create those, gerne and no genre columns
    are separated
    """
    all_dummies = pd.DataFrame()
    for column in catcolumns:
        dummies = pd.get_dummies(df[column])
        if all_dummies.empty:
            all_dummies = dummies
        else:
            all_dummies = pd.concat([all_dummies, dummies], axis=1)
    df = pd.concat([df, all_dummies], axis=1)
    return df


def _get_dummies_genre(df: pd.DataFrame, genrecolumns: list) -> pd.DataFrame:
    """
    In order to feed the model with categorical information the corresponding columns must be transformed to dummy
    variables. Since there are some differences between the columns how to create those, gerne and no genre columns
    are separated
    """
    genres = df[genrecolumns]
    genre_dummies = pd.get_dummies(genres.stack()).max(level=0)
    df = pd.concat([df, genre_dummies], axis=1)
    return df


def _descriptive_overview(df: pd.DataFrame) -> None:
    """
    Print descriptive information about the final dataset
    """
    n_users = df.cookie_id.nunique()
    print("The final dataset contains this many users:", n_users)
    n_sessions = len(df)
    print("The final dataset contains this many sessions:", n_sessions)
    n_positive = len(df[df['orders'] == 1])
    n_negative = len(df[df['orders'] == 0])
    print("The final dataset contains this many subscriptions:", n_positive)
    print("The final dataset contains this many non-subscriptions:", n_negative)
    print('Conversion rate is: {0:.2%}'.format(n_positive/(n_sessions)))


if __name__ == "__main__":
    concat_prepared_train_test()

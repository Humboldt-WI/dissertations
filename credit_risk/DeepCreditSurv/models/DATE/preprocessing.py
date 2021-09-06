import numpy as np
import pandas as pd


def f_get_Normalization(X, norm_mode):
    num_Patient, num_Feature = np.shape(X)

    if norm_mode == 'standard':  # zero mean unit variance
        for j in range(num_Feature):
            if np.std(X[:, j]) != 0:
                X[:, j] = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
            else:
                X[:, j] = (X[:, j] - np.mean(X[:, j]))
    elif norm_mode == 'normal':  # min-max normalization
        for j in range(num_Feature):
            X[:, j] = (X[:, j] - np.min(X[:, j])) / (np.max(X[:, j]) - np.min(X[:, j]))
    else:
        print("INPUT MODE ERROR!")

    return X


def formatted_data(x, t, e, idx):
    death_time = np.array(t[idx], dtype=float)
    censoring = np.array(e[idx], dtype=float)
    covariates = np.array(x[idx])

    print("observed fold:{}".format(sum(e[idx]) / len(e[idx])))
    survival_data = {'x': covariates, 't': death_time, 'e': censoring}
    return survival_data


def risk_set(data_t):
    size = len(data_t)
    risk_set = np.zeros(shape=(size, size))
    for idx in range(size):
        temp = np.zeros(shape=size)
        t_i = data_t[idx]
        at_risk = data_t > t_i
        temp[at_risk] = 1
        # temp[idx] = 0
        risk_set[idx] = temp
    return risk_set


def one_hot_encoder(data, encode):
    print("Encoding data:{}".format(data.shape))
    data_encoded = data.copy()
    encoded = pd.get_dummies(data_encoded, prefix=encode, columns=encode)
    print("head of data:{}, data shape:{}".format(data_encoded.head(), data_encoded.shape))
    print("Encoded:{}, one_hot:{}{}".format(encode, encoded.shape, encoded[0:5]))
    return encoded


def get_train_median_mode(x, categorial):
    categorical_flat = flatten_nested(categorial)
    print("categorical_flat:{}".format(categorical_flat))
    imputation_values = []
    print("len covariates:{}, categorical:{}".format(x.shape[1], len(categorical_flat)))
    median = np.nanmedian(x, axis=0)
    mode = []
    for idx in np.arange(x.shape[1]):
        a = x[:, idx]
        (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
        index = idx[np.argmax(counts)]
        mode_idx = a[index]
        mode.append(mode_idx)
    for i in np.arange(x.shape[1]):
        if i in categorical_flat:
            imputation_values.append(mode[i])
        else:
            imputation_values.append(median[i])
    print("imputation_values:{}".format(imputation_values))
    return imputation_values


def missing_proportion(dataset):
    missing = 0
    columns = np.array(dataset.columns.values)
    for column in columns:
        missing += dataset[column].isnull().sum()
    return 100 * (missing / (dataset.shape[0] * dataset.shape[1]))


def one_hot_indices(dataset, one_hot_encoder_list):
    indices_by_category = []
    for colunm in one_hot_encoder_list:
        values = dataset.filter(regex="{}_.*".format(colunm)).columns.values
        # print("values:{}".format(values, len(values)))
        indices_one_hot = []
        for value in values:
            indice = dataset.columns.get_loc(value)
            # print("column:{}, indice:{}".format(colunm, indice))
            indices_one_hot.append(indice)
        indices_by_category.append(indices_one_hot)
    # print("one_hot_indices:{}".format(indices_by_category))
    return indices_by_category


def flatten_nested(list_of_lists):
    flattened = [val for sublist in list_of_lists for val in sublist]
    return flattened


def get_missing_mask(data, imputation_values=None):
    copy = data
    for i in np.arange(len(data)):
        row = data[i]
        indices = np.isnan(row)
        # print("indices:{}, {}".format(indices, np.where(indices)))
        if imputation_values is None:
            copy[i][indices] = 0
        else:
            for idx in np.arange(len(indices)):
                if indices[idx]:
                    # print("idx:{}, imputation_values:{}".format(idx, np.array(imputation_values)[idx]))
                    copy[i][idx] = imputation_values[idx]
    # print("copy;{}".format(copy))
    return copy
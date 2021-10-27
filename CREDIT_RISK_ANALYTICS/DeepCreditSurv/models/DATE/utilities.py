import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib as mpl

from DeepCreditSurv.models.DATE.metrics import hist_plots, box_plots
from DeepCreditSurv.models.DATE.preprocessing import get_train_median_mode, formatted_data, f_get_Normalization
from sklearn.preprocessing import StandardScaler

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim

mpl.use('Agg')


def c_index(Prediction, Time_survival, Death, Time):
    '''

        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky) <class 'numpy.ndarray'>
          shape (7997,)
        - Time_survival   : survival/censoring time   <class 'numpy.ndarray'>
        shape (7997, 1)
        - Death           :   <class 'numpy.ndarray'>
            > 1: death
            > 0: censored (including death from other cause)
        shape (7997,)
        - Time            : time of evaluation (time-horizon when evaluating C-index)   <class 'int'>
        scalar value
    '''
    N = len(Prediction)  # N = 7997
    # A.shape (7997, 7997) for validation
    A = np.zeros((N, N))
    Q = np.zeros((N, N))
    N_t = np.zeros((N, N))
    Num = 0
    Den = 0
    for i in range(N):
        # np.where return the index of the values that fullfil the condition
        # let assume i = 0, then np.where(Time_survival[i] < Time_survival) will return the indexes
        # from 0 until 7996 (because the length is 7996) that have a bigger time then the index 0
        # and the assign to A[0, list of biiger times that the value in 0 ] the value 1
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1

        if (Time_survival[i] <= Time and Death[i] == 1):
            N_t[i, :] = 1

    Num = np.sum(((A) * N_t) * Q)
    Den = np.sum((A) * N_t)

    if Num == 0 and Den == 0:
        result = -1  # not able to compute c-index!
    else:
        result = float(Num / Den)

    return result


##### WEIGHTED C-INDEX & BRIER-SCORE
def CensoringProb(Y, T):
    T = T.reshape([-1])  # (N,) - np array
    Y = Y.reshape([-1])  # (N,) - np array

    kmf = KaplanMeierFitter()
    kmf.fit(T, event_observed=(Y == 0).astype(int))  # censoring prob = survival probability of event "censoring"
    G = np.asarray(kmf.survival_function_.reset_index()).transpose()
    G[1, G[1, :] == 0] = G[1, G[1, :] != 0][-1]  # fill 0 with ZoH (to prevent nan values)

    return G


def weighted_brier_score(T_train, Y_train, Prediction, T_test, Y_test, Time):
    G = CensoringProb(Y_train, T_train)
    N = len(Prediction)

    W = np.zeros(len(Y_test))
    Y_tilde = (T_test > Time).astype(float)

    for i in range(N):
        tmp_idx1 = np.where(G[0, :] >= T_test[i])[0]
        tmp_idx2 = np.where(G[0, :] >= Time)[0]

        if len(tmp_idx1) == 0:
            G1 = G[1, -1]
        else:
            G1 = G[1, tmp_idx1[0]]

        if len(tmp_idx2) == 0:
            G2 = G[1, -1]
        else:
            G2 = G[1, tmp_idx2[0]]
        W[i] = (1. - Y_tilde[i]) * float(Y_test[i]) / G1 + Y_tilde[i] / G2

    y_true = ((T_test <= Time) * Y_test).astype(float)

    return np.mean(W * (Y_tilde - (1. - Prediction)) ** 2)


def plot_predicted_distribution(predicted, empirical, data, time='days', cens=False):
    predicted_samples = np.transpose(predicted)
    print("observed_samples:{}, empirical_observed:{}".format(predicted_samples.shape,
                                                              empirical.shape))

    best_samples, diff, worst_samples = get_best_worst_indices(cens, empirical, predicted_samples)

    predicted_best = predicted_samples[best_samples]
    predicted_worst = predicted_samples[worst_samples]
    hist_plots(samples=diff, name='{}_absolute_error'.format(data), xlabel=r'|\tilde{t}-t|')

    box_plots(empirical=empirical[best_samples], predicted=list(predicted_best), name=('%s_best' % data),
              time=time)
    box_plots(empirical=empirical[worst_samples], predicted=list(predicted_worst), name=('%s_worst' % data),
              time=time)


def get_best_worst_indices(cens, empirical, predicted, size=50):
    diff = compute_relative_error(cens=cens, empirical=empirical, predicted=predicted)
    indices = sorted(range(len(abs(diff))), key=lambda k: diff[k])
    best_samples = indices[0:size]
    worst_samples = indices[len(indices) - size - 1: len(indices) - 1]
    return best_samples, diff, worst_samples


def compute_relative_error(cens, empirical, predicted, relative=False):
    predicted_median = np.median(predicted, axis=1)
    if cens:
        diff = np.minimum(0, predicted_median - empirical)
    else:
        diff = predicted_median - empirical
    if relative:
        return diff
    else:
        return abs(diff)


def batch_metrics(e, risk_set, predicted, batch_size, empirical):
    partial_likelihood = tf.constant(0.0, shape=())
    rel_abs_err = tf.constant(0.0, shape=())
    total_cens_loss = tf.constant(0.0, shape=())
    total_obs_loss = tf.constant(0.0, shape=())
    predicted = tf.squeeze(predicted)
    observed = tf.reduce_sum(e)  # compute sum across the tensor
    censored = tf.subtract(tf.cast(batch_size, dtype=tf.float32),
                           observed)  # what is left from the batch which is not bserved, it is then censored

    def condition(i, likelihood, rae, recon_loss, obs_recon_loss):
        return i < batch_size

    def body(i, likelihood, rae, cens_recon_loss, obs_recon_loss):
        # get edges for observation i
        pred_t_i = tf.gather(predicted, i)
        emp_t_i = tf.gather(empirical, i)
        e_i = tf.gather(e, i)
        censored = tf.equal(e_i, 0)
        obs_at_risk = tf.gather(risk_set, i)
        print("obs_at_risk:{}, g_theta:{}".format(obs_at_risk.shape, predicted.shape))
        risk_hazard_list = tf.multiply(predicted, obs_at_risk)
        num_adjacent = tf.reduce_sum(obs_at_risk)
        # calculate partial likelihood
        risk = tf.subtract(pred_t_i, risk_hazard_list)
        activated_risk = tf.nn.sigmoid(risk)
        # logistic = map((lambda ele: log(1 + exp(ele * -1)) * 1 / log(2)), x)
        constant = 1e-8
        log_activated_risk = tf.div(tf.log(activated_risk + constant), tf.log(2.0))
        obs_likelihood = tf.add(log_activated_risk, num_adjacent)
        uncensored_likelihood = tf.cond(censored, lambda: tf.constant(0.0), lambda: obs_likelihood)
        cumulative_likelihood = tf.reduce_sum(uncensored_likelihood)
        updated_likelihood = tf.add(cumulative_likelihood, likelihood)

        # RElative absolute error
        abs_error_i = tf.abs(tf.subtract(pred_t_i, emp_t_i))
        pred_great_empirical = tf.greater(pred_t_i, emp_t_i)
        min_rea_i = tf.minimum(tf.div(abs_error_i, pred_t_i), tf.constant(1.0))
        rea_i = tf.cond(tf.logical_and(censored, pred_great_empirical), lambda: tf.constant(0.0), lambda: min_rea_i)
        cumulative_rae = tf.add(rea_i, rae)

        # Censored generated t loss
        diff_time = tf.subtract(pred_t_i, emp_t_i)
        # logistic = map((lambda ele: log(1 + exp(ele * -1)) * 1 / log(2)), x)
        # logistic = tf.div(tf.nn.sigmoid(diff_time) + constant, tf.log(2.0))
        # hinge = map(lambda ele: max(0, 1 - ele), x)
        hinge = tf.nn.relu(1.0 - diff_time)
        censored_loss_i = tf.cond(censored, lambda: hinge, lambda: tf.constant(0.0))
        # Sum over all edges and normalize by number of edges
        # L1 recon
        observed_loss_i = tf.cond(censored, lambda: tf.constant(0.0),
                                  lambda: tf.losses.absolute_difference(labels=emp_t_i, predictions=pred_t_i))
        # add observation risk to total risk
        cum_cens_loss = tf.add(cens_recon_loss, censored_loss_i)
        cum_obs_loss = tf.add(obs_recon_loss, observed_loss_i)
        return [i + 1, tf.reshape(updated_likelihood, shape=()), tf.reshape(cumulative_rae, shape=()),
                tf.reshape(cum_cens_loss, shape=()), tf.reshape(cum_obs_loss, shape=())]

    # Relevant Functions
    idx = tf.constant(0, shape=())
    _, total_likelihood, total_rel_abs_err, batch_cens_loss, batch_obs_loss = \
        tf.while_loop(condition, body,
                      loop_vars=[idx,
                                 partial_likelihood,
                                 rel_abs_err,
                                 total_cens_loss,
                                 total_obs_loss],
                      shape_invariants=[
                          idx.get_shape(),
                          partial_likelihood.get_shape(),
                          rel_abs_err.get_shape(),
                          total_cens_loss.get_shape(),
                          total_obs_loss.get_shape()])
    square_batch_size = tf.pow(batch_size, tf.constant(2))

    def normarlize_loss(cost, size):
        cast_size = tf.cast(size, dtype=tf.float32)
        norm = tf.cond(tf.greater(cast_size, tf.constant(0.0)), lambda: tf.div(cost, cast_size), lambda: 0.0)
        return norm

    total_recon_loss = tf.add(normarlize_loss(batch_cens_loss, size=censored),
                              normarlize_loss(batch_obs_loss, size=observed))
    normalized_log_likelihood = normarlize_loss(total_likelihood, size=square_batch_size)
    return normalized_log_likelihood, normarlize_loss(total_rel_abs_err, size=batch_size), total_recon_loss


def l2_loss(scale):
    l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    return l2 * scale


def l1_loss(scale):
    l1_regularizer = tf.contrib.layers.l1_regularizer(
        scale=scale, scope=None
    )
    weights = tf.trainable_variables()  # all vars of your graph
    l1 = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
    return l1


def df_standardization(df, col_to_norm):
    x = df[col_to_norm].values
    x_scaled = StandardScaler().fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=col_to_norm, index=df.index)
    df[col_to_norm] = df_temp
    return df


def generate_data(df, time_col_name, event_col_name, col_to_norm, encoded_indices):
    # import pdb
    np.random.seed(31415)
    data_frame = df
    t_data = data_frame[time_col_name]
    e_data = data_frame[event_col_name]
    col_to_drop = [time_col_name, event_col_name]
    dataset1 = data_frame.drop(labels=col_to_drop, axis=1)
    # pdb.set_trace()

    dataset = df_standardization(dataset1, col_to_norm)

    covariates = np.array(dataset.columns.values)
    x = np.array(dataset).reshape(dataset.shape)
    t = np.array(t_data).reshape(len(t_data))
    e = np.array(e_data).reshape(len(e_data))

    idx = np.arange(0, x.shape[0])
    print("x_shape:{}".format(x.shape))

    np.random.shuffle(idx)
    x = x[idx]
    t = t[idx]
    e = e[idx]
    end_time = max(t)
    print("end_time:{}".format(end_time))
    print("observed percent:{}".format(sum(e) / len(e)))

    num_examples = int(0.80 * len(e))
    print("num_examples:{}".format(num_examples))
    vali_example = int(0.20 * num_examples)
    train_idx = idx[0: num_examples - vali_example]
    valid_idx = idx[num_examples - vali_example: num_examples]
    split = int(len(t) - num_examples)
    test_idx = idx[num_examples: num_examples + split]

    print("test:{}, valid:{}, train:{}, all: {}".format(len(test_idx), len(valid_idx), num_examples,
                                                        len(test_idx) + len(valid_idx) + num_examples))
    # print("test_idx:{}, valid_idx:{},train_idx:{} ".format(test_idx, valid_idx, train_idx))

    imputation_values = get_train_median_mode(x=np.array(x[train_idx]), categorial=encoded_indices)
    print("imputation_values:{}".format(imputation_values))
    preprocessed = {
        'train': formatted_data(x=x, t=t, e=e, idx=train_idx),
        'test': formatted_data(x=x, t=t, e=e, idx=test_idx),
        'valid': formatted_data(x=x, t=t, e=e, idx=valid_idx),
        'end_t': end_time,
        'covariates': covariates,
        'one_hot_indices': encoded_indices,
        'imputation_values': imputation_values
    }
    return preprocessed

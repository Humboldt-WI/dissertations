import numpy as np
import pandas as pd
import DeepCreditSurv.models.DeepHazard.DeepHaz as dhn
import DeepCreditSurv.models.DeepHazard.CumBaseandSurvival as cbs
import DeepCreditSurv.models.DeepHazard.concordance_index_time as cit
from pysurvival.utils._metrics import _concordance_index


def create_training_subset(inter, train, ncol, time_col_name, event_col_name):
    """ Creating all the subsets that are needed for training the model with time varying covariates
    """
    # define the subsets
    M = inter.shape[0] - 1

    T_train = train[time_col_name]
    E_train = train[event_col_name]

    Variable_Name = []
    for x in range(1, ncol + 1):
        name = 'Variable_' + "{0}".format(x)
        Variable_Name.append(name)

    subset_list = []
    X_train_list = []
    T_train_list = []
    E_train_list = []
    X_train_final_list = []
    subset1 = train.copy()
    subset1[event_col_name][subset1[time_col_name] > inter[1]] = 0
    subset1[time_col_name][subset1[time_col_name] > inter[1]] = inter[1]
    b = ["{0}".format(1)] * ncol
    ColName = [''.join(i).strip() for i in zip(Variable_Name, b)]
    X_train1 = subset1[ColName]
    X_train_final = train[ColName]
    X_train_final_list.append(X_train_final)
    T_train1 = subset1[time_col_name]
    E_train1 = subset1[event_col_name]
    subset_list.append(subset1)
    X_train_list.append(X_train1)
    T_train_list.append(T_train1)
    E_train_list.append(E_train1)

    for x in range(2, M + 1):
        train_temp = train.copy()
        subset = train_temp[train_temp[time_col_name] > inter[x - 1]]
        subset[event_col_name][subset[time_col_name] > inter[x]] = 0
        subset[time_col_name][subset[time_col_name] > inter[x]] = inter[x]
        b = ["{0}".format(x)] * ncol
        ColName = [''.join(i).strip() for i in zip(Variable_Name, b)]
        X_train = subset[ColName]
        T_train2 = subset[time_col_name]
        E_train2 = subset[event_col_name]
        X_train_final = train[ColName]
        X_train_final_list.append(X_train_final)
        X_train_int_list = []
        X_train_int_list.append(X_train)
        T_train_list.append(T_train2)
        E_train_list.append(E_train2)
        for i in range(1, x):
            b = ["{0}".format(i)] * ncol
            ColName = [''.join(i).strip() for i in zip(Variable_Name, b)]
            X_train2 = subset[ColName]
            X_train_int_list.append(X_train2)
        X_train_list.append(X_train_int_list)
    return X_train_list, T_train_list, E_train_list, X_train_final_list, T_train, E_train


def create_test_subset(inter, test, ncol, time_col_name, event_col_name):
    """ Creating all the subsets that are needed for applying the model with time varying covariates
    """
    # define the subsets
    M = inter.shape[0] - 1

    T_test = test[time_col_name]
    E_test = test[event_col_name]

    Variable_Name = []
    for x in range(1, ncol + 1):
        name = 'Variable_' + "{0}".format(x)
        Variable_Name.append(name)

    X_test_list = []
    for x in range(1, M + 1):
        b = ["{0}".format(x)] * ncol
        ColName = [''.join(i).strip() for i in zip(Variable_Name, b)]
        X_test = test[ColName]
        X_test_list.append(X_test)

    T_test = test[time_col_name]
    E_test = test[event_col_name]

    return X_test_list, T_test, E_test


def train_deephaz_time(train, inter, ncol, l2c, lrc, structure, init_method, optimizer, num_epochs, early_stopping,
                       penal, time_col_name, event_col_name):
    """ Training the model with time_varying covariates
    """
    X_train_list, T_train_list, E_train_list, X_train_final_list, T_train, E_train = create_training_subset(inter,
                                                                                                            train,
                                                                                                            ncol,
                                                                                                            time_col_name,
                                                                                                            event_col_name)
    M = inter.shape[0] - 1
    deephaz_lis = []
    Ttemp = T_train_list[0]
    Etemp = E_train_list[0]
    Xtemp = X_train_list[0]
    deephaz1 = dhn.DeepHaz(structure=structure)
    deephaz1.fit(Xtemp, Ttemp, Etemp, lr=lrc, init_method=init_method, optimizer=optimizer, num_epochs=num_epochs,
                 l2_reg=l2c, early_stopping=early_stopping, penal=penal)
    deephaz_lis.append(deephaz1)
    score_list = []
    score1 = deephaz1.predict_risk(X_train_final_list[0])
    score1.shape = (score1.shape[0], 1)
    score_list.append(score1)
    scoretemp = deephaz1.predict_risk(X_train_list[1][1])
    scoretemp.shape = (scoretemp.shape[0], 1)
    X_train2n = np.concatenate((X_train_list[1][0], scoretemp), 1)
    for x in range(2, M):
        Ttemp = T_train_list[x - 1]
        Etemp = E_train_list[x - 1]
        deephaz2 = dhn.DeepHaz(structure=structure)
        deephaz2.fit(X_train2n, Ttemp, Etemp, lr=lrc, t_start=inter[x - 1], init_method='he_uniform', optimizer='adam',
                     num_epochs=1000, l2_reg=l2c, early_stopping=1e-5, penal='Ridge')
        deephaz_lis.append(deephaz2)
        trainscore = X_train_final_list[x - 1]
        for i in range(x - 2, -1, -1):
            trainscore = np.concatenate((trainscore, score_list[i]), 1)
        score2 = deephaz2.predict_risk(trainscore)
        score2.shape = (score2.shape[0], 1)
        score_list.append(score2)
        score_temp = []
        for i in range(x):
            trainscore = X_train_list[x][i + 1]
            for j in score_temp[::-1]:
                trainscore = np.concatenate((trainscore, j), 1)
            score31 = deephaz_lis[i].predict_risk(trainscore)
            score31.shape = (score31.shape[0], 1)
            score_temp.append(score31)
        X_train2n = X_train_list[x][0]
        for j in score_temp[::-1]:
            X_train2n = np.concatenate((X_train2n, j), 1)
    Ttemp = T_train_list[M - 1]
    Etemp = E_train_list[M - 1]
    deephaz2 = dhn.DeepHaz(structure=structure)
    deephaz2.fit(X_train2n, Ttemp, Etemp, lr=lrc, t_start=inter[x - 1], init_method='he_uniform', optimizer='adam',
                 num_epochs=1000, l2_reg=l2c, early_stopping=1e-5, penal='Ridge')
    deephaz_lis.append(deephaz2)
    trainscore = X_train_final_list[M - 1]
    for i in range(M - 2, -1, -1):
        trainscore = np.concatenate((trainscore, score_list[i]), 1)
    score2 = deephaz2.predict_risk(trainscore)
    score2.shape = (score2.shape[0], 1)
    score_list.append(score2)

    score = score_list[0].reshape((-1, 1))
    for j in range(1, M):
        score = np.concatenate((score, score_list[j].reshape((-1, 1))), axis=1)

    indicator = list(range(T_train.shape[0]))
    for i in range(T_train.shape[0]):
        if T_train[i] < inter[1]:
            indicator[i] = 1
    for i in range(T_train.shape[0]):
        for j in range(1, inter.shape[0] - 1):
            if T_train[i] < inter[j + 1] and T_train[i] >= inter[j]:
                indicator[i] = j + 1

    cumbase = cbs.predict_cumbase(score, T_train, E_train, inter, indicator)

    time = T_train

    return deephaz_lis, score, cumbase, time


def predict_deephaz_time(inter, test, ncol, deephaz_lis, cumbase, time, time_col_name, event_col_name):
    """ Use the model to predict survival function with time varying covariates
    """
    M = inter.shape[0] - 1
    score_list = []
    X_test_list, T_test, E_test = create_test_subset(inter, test, ncol, time_col_name, event_col_name)
    score1 = deephaz_lis[0].predict_risk(X_test_list[0])
    score1.shape = (score1.shape[0], 1)
    score_list.append(score1)
    for i in range(1, M):
        testscore = X_test_list[i]
        for j in score_list[::-1]:
            testscore = np.concatenate((testscore, j), 1)
        score1 = deephaz_lis[i].predict_risk(testscore)
        score1.shape = (score1.shape[0], 1)
        score_list.append(score1)
    score = score_list[0].reshape((-1, 1))
    for j in range(1, M):
        score = np.concatenate((score, score_list[j].reshape((-1, 1))), axis=1)

    indicator = list(range(time.shape[0]))
    for i in range(time.shape[0]):
        if time[i] < inter[1]:
            indicator[i] = 1
    for i in range(time.shape[0]):
        for j in range(1, inter.shape[0] - 1):
            if inter[j + 1] > time[i] >= inter[j]:
                indicator[i] = j + 1

    Surv = cbs.predict_surv(cumbase, score, time, inter, indicator, use_log=False)

    return score, Surv


def deephaz_time(train, test, inter, ncol, l2c, lrc, structure, init_method, optimizer, num_epochs, early_stopping,
                 penal, time_col_name, event_col_name):
    """ Training deep_hazard on training data with time varying covariates and use the model to predict the Survival function onto a test dataset.
    Parameters:
        -----------
        * train : pandas dataframe that contains the training data. Time column needs to be called 'Time',Event indicator column needs to be called 'Event', 
                  Variables need to be called : 'Variable_ij' where i is the number of the variable
                  j is the interval onto which the variable gets that value
        * test : pandas dataframe that contains the test data. Time columns needs to be called 'Time',Event indicator columns needs to be called 'Event', 
                  Variables need to be called : 'Variable_ij' where i is the number of the variable
                  j is the interval onto which the variable gets that value
        * inter : np.array with the extremes of the intervals. 
        * ncol : Dimesion of covariates
        * lrc : **float** *(default=1e-4)* -- 
            learning rate used in the optimization
        * l2c : **float** *(default=1e-4)* -- 
            regularization parameter for the model coefficients
        * structure: List of dictionaries
                ex: structure = [ {'activation': 'relu', 'num_units': 128,'dropout':0.2}, 
                                  {'activation': 'tanh', 'num_units': 128,'dropout':0.2}, ] 
        * init_method : **str** *(default = 'glorot_uniform')* -- 
            Initialization method to use. Here are the possible options:
            * `glorot_uniform`: Glorot/Xavier uniform initializer
            * `he_uniform`: He uniform variance scaling initializer 
            * `uniform`: Initializing tensors with uniform (-1, 1) distribution
            * `glorot_normal`: Glorot normal initializer,
            * `he_normal`: He normal initializer.
            * `normal`: Initializing tensors with standard normal distribution
            * `ones`: Initializing tensors to 1
            * `zeros`: Initializing tensors to 0
            * `orthogonal`: Initializing tensors with a orthogonal matrix,
        * optimizer :  **str** *(default = 'adam')* -- 
            iterative method for optimizing a differentiable objective function.
            Here are the possible options:
            - `adadelta`
            - `adagrad`
            - `adam`
            - `adamax`
            - `rmsprop`
            - `sparseadam`
            - `sgd`
        * num_epochs: **int** *(default=1000)* -- 
            The number of iterations in the optimization
        * early_stopping: early stopping tolerance
        * penal: 'Ridge' if we want to apply Ridge penalty to the loss
                 'Lasso' if we want to apply Lasso penalty to the loss        
    Outputs:
        -----------
        * deepHazlis : A list that contains the trained networks
        * Surv: np.array with the predicted survival, each rows correspond to a different observation in the test set
                each column correspond to different times (times in the training data)
        * Time dependent concorance index from
           Antolini, L.; Boracchi, P.; and Biganzoli, E. 2005. A timedependent discrimination
           index for survival data. Statistics in Medicine 24:3927–3944.
  
    """
    deephaz_lis, score, cumbase, time = train_deephaz_time(train, inter, ncol, l2c, lrc, structure, init_method,
                                                           optimizer, num_epochs, early_stopping, penal, time_col_name,
                                                           event_col_name)
    score, Surv = predict_deephaz_time(inter, test, ncol, deephaz_lis, cumbase, time, time_col_name, event_col_name)
    X_test_list, T_test, E_test = create_test_subset(inter, test, ncol, time_col_name, event_col_name)
    T_test = np.array(T_test)
    E_test = np.array(E_test)
    C_index = cit.concordance_td(T_test, E_test, np.transpose(Surv), np.arange(T_test.shape[0]), method='antolini')
    return deephaz_lis, Surv, C_index


def deephaz_const(train, test, l2c, lrc, structure, init_method, optimizer,
                  num_epochs, early_stopping, penal, time_col_name, event_col_name):
    """ Training deep_hazard on training data with time varying covariates and use the model to predict the Survival function onto a test dataset.
    Parameters:
        -----------
        * train : pandas dataframe that contains the training data. Time column needs to be called 'Time',Event indicator column needs to be called 'Event', 
        * test : pandas dataframe that contains the test data. Time columns needs to be called 'Time',Event indicator columns needs to be called 'Event',
        * lrc : **float** *(default=1e-4)* -- 
            learning rate used in the optimization
        * l2c : **float** *(default=1e-4)* -- 
            regularization parameter for the model coefficients
        * structure: List of dictionaries
                ex: structure = [ {'activation': 'relu', 'num_units': 128,'dropout':0.2}, 
                                  {'activation': 'tanh', 'num_units': 128,'dropout':0.2}, ] 
        * init_method : **str** *(default = 'glorot_uniform')* -- 
            Initialization method to use. Here are the possible options:
            * `glorot_uniform`: Glorot/Xavier uniform initializer
            * `he_uniform`: He uniform variance scaling initializer 
            * `uniform`: Initializing tensors with uniform (-1, 1) distribution
            * `glorot_normal`: Glorot normal initializer,
            * `he_normal`: He normal initializer.
            * `normal`: Initializing tensors with standard normal distribution
            * `ones`: Initializing tensors to 1
            * `zeros`: Initializing tensors to 0
            * `orthogonal`: Initializing tensors with a orthogonal matrix,
        * optimizer :  **str** *(default = 'adam')* -- 
            iterative method for optimizing a differentiable objective function.
            Here are the possible options:
            - `adadelta`
            - `adagrad`
            - `adam`
            - `adamax`
            - `rmsprop`
            - `sparseadam`
            - `sgd`
        * num_epochs: **int** *(default=1000)* -- 
            The number of iterations in the optimization
        * early_stopping: early stopping tolerance
        * penal: 'Ridge' if we want to apply Ridge penalty to the loss
                 'Lasso' if we want to apply Lasso penalty to the loss
        * time_col_name: name of the time data column
        * event_col_name: name of the event data column
    Outputs:
        -----------
        * deepHaz : The trained Network
        * Surv: np.array with the predicted survival, each rows correspond to a different observation in the test set
                each column correspond to different times (times in the training data)
        * C_index
    """

    T_train = train[time_col_name]
    E_train = train[event_col_name]
    X_train = train.copy()
    X_train = X_train.drop([time_col_name, event_col_name], axis=1)
    T_test = test[time_col_name]
    E_test = test[event_col_name]
    X_test = test.copy()
    X_test = X_test.drop([time_col_name, event_col_name], axis=1)
    deephaz = dhn.DeepHaz(structure=structure)
    deephaz.fit(X_train, T_train, E_train, lr=lrc, init_method=init_method, optimizer=optimizer, num_epochs=num_epochs,
                l2_reg=l2c, early_stopping=early_stopping, penal=penal)
    score = deephaz.predict_risk(X_test, use_log=False)
    # order = np.argsort(-T_test)
    # score = score[order]
    # T_test = T_test[order]
    # E_test = E_test[order]
    C_index = _concordance_index(score, T_test, E_test, True)[0]

    return deephaz, score, C_index

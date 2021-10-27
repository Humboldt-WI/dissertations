# results table
CREATE TABLE results
(
    id             INTEGER                             PRIMARY KEY AUTOINCREMENT,
    test_dataset   TEXT(1000)                          NOT NULL,
    train_dataset  TEXT(1000)                          NOT NULL,
    iteration      INTEGER                             NOT NULL,
    train_size     INTEGER                             NOT NULL,
    gamma          REAL                                NOT NULL,
    attention_func TEXT(1000)                          NOT NULL,
    created        TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL,
    sentence       TEXT(1000)                          NOT NULL,
    y_true         TEXT(1000)                          NOT NULL,
    y_pred         TEXT(1000)                          NOT NULL,
    UNIQUE (test_dataset, train_dataset, iteration, train_size, gamma, attention_func, sentence)
);



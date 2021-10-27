import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier


class Split:
    def __init__(self, csv_path):
        self.df = self.load_and_process(csv_path)

    def load_and_process(self, csv_path):
        df = pd.read_csv(csv_path)
        df = df.fillna(0)
        df = df.dropna()
        return df 

    def run_experiment(self, path_to_save, ignore_label):
        results = {
            "dataset": [],
            "ratio": [], 
            "model": [],
            "fold1": [],
            "fold2": [],
            "fold3": [],
            "fold4": [],
            "mean": [],
            "test": []
        }

        for i, df_train in enumerate([self.df]):

            df_train = df_train[df_train["Label"]!=ignore_label]
            df_train = df_train.reset_index(drop=True)
            
            for ratio in [0.25, 0.5, 0.75]:
                print(f"Training on dataset {i+1} - ratio {ratio}")
                encoder = LabelEncoder()
                X = df_train.drop(["Label", "TimeID", "Date", "Name", "Hour", "Minute"], axis=1)
                X = np.array(X)
                y = encoder.fit_transform(df_train["Label"])
                
                split_idx = int(len(X)*ratio)

                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                rf = RandomForestClassifier(random_state=18)
                xgb = XGBClassifier(random_state=18)
                svc = Pipeline([('standardize', StandardScaler()), ('svc', SVC())])
                logistic = LogisticRegression(solver="saga", multi_class="ovr", max_iter=200, tol=1e-2)
                mlp = Pipeline(steps=[('normalize', MinMaxScaler()), ('mlp', MLPClassifier(max_iter=500, random_state=18))])
                models = [rf, xgb, svc, logistic, mlp]
                names = ["RF", "XGB", "SVC", "Logistic", "MLP"]
                
                for model, name in zip(models, names):
                    scores = evaluate(model, X_train, y_train)
                    model.fit(X_train, y_train)
                    test_score = model.score(X_test, y_test)
                    print(f"{name} CV results: {scores}, Mean: {np.mean(scores)}")
                    print(f"{name} test result: {test_score}", )
                    print()
                    
                    results["dataset"].append(i+1)
                    results["ratio"].append(ratio)
                    results["model"].append(name)
                    results["fold1"].append(scores[0])
                    results["fold2"].append(scores[1])
                    results["fold3"].append(scores[2])
                    results["fold4"].append(scores[3])
                    results["mean"].append(np.mean(scores))
                    results["test"].append(test_score)

        df_results = pd.DataFrame(results)
        df_results.to_csv(path_to_save, index=False)

    
def evaluate(model, inputs, labels, n_fold=4):
        cross_validation_set = KFold(n_splits=n_fold)
        scores = cross_val_score(model, inputs, labels, cv=cross_validation_set)
        return scores
    
if __name__ == "__main__":
    for i, ignore in zip([1, 2], [7, 10]):
        base = f"./data/Deployment_{i}/"
        for type in ["mean", "max"]:
            path_to_csv = base + f"dataset_{i}_preprocessing_with_{type}_value.csv"
            S = Split(path_to_csv)
            S.run_experiment(base + f"expriment_deployment_{i}_{type}.csv", ignore)
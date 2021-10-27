import pandas as pd
from src.utils.processing import split_timestamp, creat_dataset

class Dataset:
    def __init__(self, base):
        self.base = base

    def preprocess_SensorHistory(self, path_to_save=''):
        sensor_history = pd.read_csv(self.base + "SensorEventHistory.csv")
        sensor_history = split_timestamp(sensor_history)
        sensor_history = sensor_history[["Date", "Hour", "Minute", "Value", "SensorID", "TimeID"]]
        if path_to_save:
            sensor_history.to_csv(path_to_save, index=False)

    def make_dataset(self, cols_to_include, path_to_preprocessed_SensorHistory, path_to_save, start, end, process_with_mean=True):
        df_act = pd.read_csv(self.base + "Activities.csv")
        df_act = split_timestamp(df_act)
        df_act, _ = creat_dataset(start, end, df_act)

        sensor_history = pd.read_csv(path_to_preprocessed_SensorHistory)
        sensors = pd.read_csv(self.base + "Sensors.csv")

        for sensor_id in sensors["SensorID"]:
            if sensor_id in cols_to_include:
                
                sensor_tmp = sensor_history[sensor_history["SensorID"]==sensor_id]
                sensor_tmp = sensor_tmp.rename(columns={"Value":f"sensor{sensor_id}"})
                sensor_tmp = sensor_tmp[[f"sensor{sensor_id}", "TimeID"]]
                sensor_tmp = sensor_tmp.groupby("TimeID")
                if process_with_mean:
                    sensor_tmp = sensor_tmp.mean()
                else:
                    sensor_tmp = sensor_tmp.max()
                sensor_tmp = sensor_tmp.reset_index()
                df_act = pd.merge(df_act, sensor_tmp, how="left", on="TimeID")
        df_act.to_csv(path_to_save, index=False)
        
if __name__ == "__main__":
    cols_1 = [1,2,3,4,5,6,8,9,11,12,20,21,22,23,24,25,30,31,32,34,35,40]
    cols_2 = [1,2,3,5,6,20,21,22,24,30,31,32,40,41,4,42]
    cols = [cols_1, cols_2]
    starts = ["2013-03-04", "2013-04-14"]
    ends = ["2013-06-11", "2013-06-16"]
    for i, col, start, end in zip([1, 2], cols, starts, ends):
        base = f"./data/Deployment_{i}/"
        preprocessed_SensorHistory_path = base + f"Deployment_{i}_sensor_preprocessing_data.csv"
        
        Data = Dataset(base)
        Data.preprocess_SensorHistory(preprocessed_SensorHistory_path)
        
        for process_with_mean in [True, False]:
            path_to_save = base + f"dataset_{i}_preprocessing_with_max_value.csv"
            if process_with_mean:
                path_to_save = base + f"dataset_{i}_preprocessing_with_mean_value.csv"
            Data.make_dataset(col, preprocessed_SensorHistory_path, path_to_save, start, end, process_with_mean)

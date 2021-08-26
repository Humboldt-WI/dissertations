import pandas as pd

def split_timestamp(df):
    """
    split TimeStamp attribute into 4 columns 
    including Date, Hour, Minute, TimeID

    Args:
        df : a dataframe

    Returns:
        dataframe
    """
    time_stamp = df["TimeStamp"].tolist()
    dates = [x.split()[0] for x in time_stamp]
    clock = [x.split()[1].split(":") for x in time_stamp]
    hh = [x[0] for x in clock]
    mm = [x[1] for x in clock]
    time_id = ["-".join([d, h, m]) for d,h,m in zip(dates, hh, mm)]
    df["Date"] = dates
    df["Hour"] = list(map(int, hh))
    df["Minute"] = list(map(int, mm))
    df["TimeID"] = time_id
    return df

def creat_dataset(start, end, df_activities):
    """
    create a dataset with 1-minute step

    Args:
        start : starting datatime
        end : ending datatime
        df_activities : activities dataframe

    Returns:
        dataset
    """
    #Initialize a time interval with minutely frequency
    time_interval = pd.date_range(start=start, end=end, freq="T")
    time_stamp = [str(time_point) for time_point in time_interval]
    
    #Create a dataframe with TimeStamp
    df_time = pd.DataFrame({"TimeStamp": time_stamp})
    df_time = split_timestamp(df_time)
    #df_time = df_time[["TimeID"]]
    
    assert "TimeID" in df_activities.columns, "Check if Activities has column TimeID"
    
    #Merge TimeStamp with activities 
    df_time_merge = pd.merge(df_time, df_activities[["TimeID", "Value", "Name"]], how="left", on="TimeID")
    
    #Filter out time interval beyond timestamp of activities df
    TimeID_start = df_activities.loc[0, "TimeID"]
    TimeID_stop = df_activities.loc[len(df_activities)-1, "TimeID"]
    row_start = df_time_merge[df_time_merge["TimeID"]==TimeID_start].index.item()
    row_stop = df_time_merge[df_time_merge["TimeID"]==TimeID_stop].index.item()
    df_time_final = df_time_merge[row_start:row_stop+1]
    df_time_final = df_time_final.reset_index(drop=True)
    
    df_time_final = df_time_final[["TimeID", "Hour", "Minute", "Date", "Value", "Name"]]
    df_return = df_time_final = df_time_final.rename(columns={"Value": "Label"})
    
    #Fill label column with previous values
    df_time_final = df_time_final.fillna(method="ffill")
    
    #Drop duplicates
    df_time_final = df_time_final.drop_duplicates()
    
    return df_time_final, df_return
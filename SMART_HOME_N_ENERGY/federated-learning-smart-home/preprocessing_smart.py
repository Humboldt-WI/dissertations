
### Preprocessing
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("fivethirtyeight")
plt.rcParams['figure.figsize'] = (25, 15)

path="/Users/daniel.bustillo/Documents/thesis"

os.chdir(path)

#Reading all years from House A

home_a_1_15= pd.read_csv("Dataset/HomeA/2015/HomeA-meter2_2015.csv",infer_datetime_format=True, index_col=0, parse_dates=True)
home_a_2_15= pd.read_csv("Dataset/HomeA/2015/HomeA-meter3_2015.csv",infer_datetime_format=True, index_col=0, parse_dates=True)

home_a_1_16= pd.read_csv("Dataset/HomeA/2016/HomeA-meter2_2016.csv",infer_datetime_format=True, index_col=0, parse_dates=True)
home_a_2_16= pd.read_csv("Dataset/HomeA/2016/HomeA-meter3_2016.csv",infer_datetime_format=True, index_col=0, parse_dates=True)


#Reading all years from House B

home_b_1 = pd.read_csv("Dataset/HomeB/2014/HomeB-meter1_2014.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_b_2 = pd.read_csv("Dataset/HomeB/2014/HomeB-meter2_2014.csv", infer_datetime_format=True, index_col=0, parse_dates=True)

home_b_1_15 = pd.read_csv("Dataset/HomeB/2015/HomeB-meter1_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_b_2_15 = pd.read_csv("Dataset/HomeB/2015/HomeB-meter2_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)


home_b_1_16 = pd.read_csv("Dataset/HomeB/2015/HomeB-meter1_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_b_2_16 = pd.read_csv("Dataset/HomeB/2015/HomeB-meter2_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)





#Reading all years from House C
home_c_1 = pd.read_csv("Dataset/HomeC/2014/HomeC-meter1_2014.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_c_2 = pd.read_csv("Dataset/HomeC/2014/HomeC-meter2_2014.csv", infer_datetime_format=True, index_col=0, parse_dates=True)

home_c_1_15 = pd.read_csv("Dataset/HomeC/2015/HomeC-meter1_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_c_2_15 = pd.read_csv("Dataset/HomeC/2015/HomeC-meter2_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)

home_c_1_16 = pd.read_csv("Dataset/HomeC/2016/HomeC-meter1_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_c_2_16 = pd.read_csv("Dataset/HomeC/2016/HomeC-meter2_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)


# Reading all years from House D
home_d_1 = pd.read_csv("Dataset/HomeD/2015/HomeD-meter1_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_d_2 = pd.read_csv("Dataset/HomeD/2015/HomeD-meter2_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_d_3 = pd.read_csv("Dataset/HomeD/2015/HomeD-meter3_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_d_4 = pd.read_csv("Dataset/HomeD/2015/HomeD-meter4_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)

home_d_1_16 = pd.read_csv("Dataset/HomeD/2016/HomeD-meter1_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_d_2_16 = pd.read_csv("Dataset/HomeD/2016/HomeD-meter2_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_d_3_16 = pd.read_csv("Dataset/HomeD/2016/HomeD-meter3_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_d_4_16 = pd.read_csv("Dataset/HomeD/2016/HomeD-meter4_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)


#Reading all years from House F

home_f_2 = pd.read_csv("Dataset/HomeF/2014/HomeF-meter2_2014.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_f_3 = pd.read_csv("Dataset/HomeF/2014/HomeF-meter3_2014.csv", infer_datetime_format=True, index_col=0, parse_dates=True)


home_f_2_15 = pd.read_csv("Dataset/HomeF/2015/HomeF-meter2_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_f_3_15 = pd.read_csv("Dataset/HomeF/2015/HomeF-meter3_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)


home_f_2_16 = pd.read_csv("Dataset/HomeF/2016/HomeF-meter2_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_f_3_16 = pd.read_csv("Dataset/HomeF/2016/HomeF-meter3_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True) 

# Reading all years from House G
home_g_1 = pd.read_csv("Dataset/HomeG/2015/HomeG-meter1_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_g_2 = pd.read_csv("Dataset/HomeG/2015/HomeG-meter2_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_g_4 = pd.read_csv("Dataset/HomeG/2015/HomeG-meter4_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_g_5 = pd.read_csv("Dataset/HomeG/2015/HomeG-meter5_2015.csv", infer_datetime_format=True, index_col=0, parse_dates=True)

home_g_1_16 = pd.read_csv("Dataset/HomeG/2016/HomeG-meter1_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_g_2_16 = pd.read_csv("Dataset/HomeG/2016/HomeG-meter2_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_g_4_16 = pd.read_csv("Dataset/HomeG/2016/HomeG-meter4_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_g_5_16 = pd.read_csv("Dataset/HomeG/2016/HomeG-meter5_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)

# Reading all years from House H
home_h_1 = pd.read_csv("Dataset/HomeH/2016/HomeH-meter1_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)
home_h_2 = pd.read_csv("Dataset/HomeH/2016/HomeH-meter2_2016.csv", infer_datetime_format=True, index_col=0, parse_dates=True)


home_a_1_15 = home_a_1_15.resample("1H").mean()
home_a_2_15= home_a_2_15.resample("1H").mean()
home_a_1_16=home_a_1_16.resample("1H").mean()
home_a_2_16 = home_a_2_16.resample("1H").mean()

home_b_1 = home_b_1.resample("1H").mean()
home_b_2 = home_b_2.resample("1H").mean()
home_b_1_15 = home_b_1_15.resample("1H").mean()
home_b_2_15 = home_b_2_15.resample("1H").mean()
home_b_1_16 = home_b_1_16.resample("1H").mean()
home_b_2_16 = home_b_2_16.resample("1H").mean()


home_c_1 = home_c_1.resample("1H").mean()
home_c_2 = home_c_2.resample("1H").mean()
home_c_1_15 = home_c_1_15.resample("1H").mean()
home_c_2_15 = home_c_2_15.resample("1H").mean()
home_c_1_16 = home_c_1_16.resample("1H").mean()
home_c_2_16 = home_c_2_16.resample("1H").mean()


home_d_1 = home_d_1.resample("1H").mean()
home_d_2 = home_d_2.resample("1H").mean()
home_d_3 = home_d_3.resample("1H").mean()
home_d_4 = home_d_4.resample("1H").mean()

home_d_1_16 = home_d_1_16.resample("1H").mean()
home_d_2_16 = home_d_2_16.resample("1H").mean()
home_d_3_16 = home_d_3_16.resample("1H").mean()
home_d_4_16 = home_d_4_16.resample("1H").mean()

home_f_2 =home_f_2.resample("1H").mean()
home_f_3 = home_f_3.resample("1H").mean()

home_f_2_15 = home_f_2_15.resample("1H").mean()
home_f_3_15 = home_f_3_15.resample("1H").mean()

home_f_2_16 = home_f_2_16.resample("1H").mean()
home_f_3_16 = home_f_3_16.resample("1H").mean()

home_g_1 = home_g_1.resample("1H").mean()
home_g_2 = home_g_2.resample("1H").mean()
home_g_4 = home_g_4.resample("1H").mean()
home_g_5 = home_g_5.resample("1H").mean()

home_g_1_16 = home_g_1_16.resample("1H").mean()
home_g_2_16 = home_g_2_16.resample("1H").mean()
home_g_4_16 = home_g_4_16.resample("1H").mean()
home_g_5_16 = home_g_5_16.resample("1H").mean()


home_h_1 = home_h_1.resample("1H").mean()
home_h_2 = home_h_2.resample("1H").mean()

home_d_1.columns.intersection(home_d_2.columns)


home_c_1 = home_c_1.drop(['gen [kW]','House overall [kW]','Barn [kW]', 'Well [kW]','Microwave [kW]', 'Solar [kW]'], axis=1)

home_c_1_15= home_c_1_15.drop(['gen [kW]','House overall [kW]','Barn [kW]', 'Well [kW]','Microwave [kW]', 'Solar [kW]'], axis=1)

home_c_1_16 = home_c_1_16.drop(['gen [kW]','House overall [kW]','Barn [kW]', 'Well [kW]','Microwave [kW]', 'Solar [kW]'], axis=1)

home_d_2 = home_d_2.drop(['use [kW]', 'gen [kW]', 'SecondFloorBathroom [kW]',
       'GuestHouseKitchen [kW]', 'GuestHouseKitchen [kW].1',
       'GroundSourceHeatPump [kW]', 'Photovoltaics [kW]', 'WellPump [kW]',
       'Range [kW]', 'PanelReceptacles [kW]', 'KitchenReceptacles [kW]',
       'Microwave [kW]', 'Refrigerator [kW]'], axis=1)


home_d_2_16 = home_d_2_16.drop(['use [kW]', 'gen [kW]', 'SecondFloorBathroom [kW]',
       'GuestHouseKitchen [kW]', 'GuestHouseKitchen [kW].1',
       'GroundSourceHeatPump [kW]', 'Photovoltaics [kW]', 'WellPump [kW]',
       'Range [kW]', 'PanelReceptacles [kW]', 'KitchenReceptacles [kW]',
       'Microwave [kW]', 'Refrigerator [kW]'], axis=1)

home_f_2 = home_f_2.drop(['Usage [kW]', 'Generation [kW]', 'Solar [kW]', 'Phase_B [kW]',
       'Phase_A [kW]'], axis=1)
       

home_f_2_15 = home_f_2_15.drop(['Usage [kW]', 'Generation [kW]', 'Solar [kW]', 'Phase_B [kW]',
       'Phase_A [kW]'], axis=1)

home_f_2_16 = home_f_2_16.drop(['Usage [kW]', 'Generation [kW]', 'Solar [kW]', 'Phase_B [kW]',
       'Phase_A [kW]'], axis=1)

home_g_2 = home_g_2.drop(['use [kW]', 'gen [kW]'], axis=1)
home_g_4 = home_g_4.drop(['use [kW]', 'gen [kW]'], axis=1)
home_g_5 = home_g_5.drop(['use [kW]', 'gen [kW]'], axis=1)

home_g_2_16 = home_g_2_16.drop(['use [kW]', 'gen [kW]'], axis=1)
home_g_4_16 = home_g_4_16.drop(['use [kW]', 'gen [kW]'], axis=1)
home_g_5_16 = home_g_5_16.drop(['use [kW]', 'gen [kW]'], axis=1)

home_h_2 = home_h_2.drop(['Usage [kW]', 'Generation [kW]'],axis=1)

#home_a_2 =home_a_2.drop("use [kW]", axis=1)
#home_a= home_a_1.merge(right= home_a_2, how="outer",left_index= True, right_index= True)

home_a_2_15 =home_a_2_15.drop("use [kW]", axis=1)
home_a_15= home_a_1_15.merge(right= home_a_2_15, how="outer",left_index= True, right_index= True)

home_a_2_16 =home_a_2_16.drop("use [kW]", axis=1)
home_a_16= home_a_1_16.merge(right= home_a_2_16, how="outer",left_index= True, right_index= True)


# dfs=[home_a, home_a_15, home_a_16]
dfs=[home_a_15, home_a_16]

home_a= pd.concat(dfs)

del home_a_1_15, home_a_2_15, home_a_1_16, home_a_2_16, home_a_15, home_a_16

home_b_2= home_b_2.drop('use [kW]', axis=1)
home_b= home_b_1.merge(right=home_b_2, how='outer', left_index=True, right_index=True)

home_b_2_15= home_b_2_15.drop('use [kW]', axis=1)
home_b_15= home_b_1_15.merge(right=home_b_2_15, how='outer', left_index=True, right_index=True)

home_b_2_16= home_b_2_16.drop('use [kW]', axis=1)
home_b_16= home_b_1_16.merge(right=home_b_2_16, how='outer', left_index=True, right_index=True)

dfs=[home_b, home_b_15, home_b_16]

home_b = pd.concat(dfs)

del home_b_1, home_b_2, home_b_1_15, home_b_2_15, home_b_1_16, home_b_2_16, home_b_15, home_b_16

home_c_2= home_c_2.drop('use [kW]', axis=1)
home_c= home_c_1.merge(right=home_c_2, how='outer', left_index=True, right_index=True)

home_c_2_15= home_c_2_15.drop('use [kW]', axis=1)
home_c_15= home_c_1_15.merge(right=home_c_2_15, how='outer', left_index=True, right_index=True)

home_c_2_16= home_c_2_16.drop('use [kW]', axis=1)
home_c_16= home_c_1_16.merge(right=home_c_2_16, how='outer', left_index=True, right_index=True)

dfs=[home_c, home_c_15, home_c_16]

home_c = pd.concat(dfs)

del home_c_1, home_c_2, home_c_1_15, home_c_2_15, home_c_1_16, home_c_2_16, home_c_15, home_c_16

home_d1 = home_d_1.merge(right=home_d_2, how='outer', left_index=True, right_index=True)
home_d1_16 = home_d_1_16.merge(right= home_d_2_16, how='outer', left_index=True, right_index=True)

dfs = [home_d1, home_d1_16]
home_d = pd.concat(dfs)

del home_d_1, home_d_2, home_d_3, home_d_4, home_d_1_16, home_d_2_16, home_d_3_16, home_d_4_16, home_d1, home_d1_16

home_f_1 = home_f_2.merge(right=home_f_3, how="outer", left_index=True, right_index=True)
home_f_1_15 = home_f_2_15.merge(right=home_f_3_15, how="outer", left_index=True, right_index=True)
home_f_1_16 = home_f_2_16.merge(right=home_f_3_16, how="outer", left_index=True, right_index=True)

dfs=[home_f_1, home_f_1_15, home_f_1_16]
home_f = pd.concat(dfs)

del home_f_2, home_f_3, home_f_2_15, home_f_3_15, home_f_2_16, home_f_3_16, home_f_1, home_f_1_15, home_f_1_16

home_g_c = home_g_1.merge(right=home_g_2, how='outer', left_index=True, right_index=True)
home_g_c1 = home_g_c.merge(right=home_g_4, how='outer', left_index=True, right_index=True)
home_g_c2 = home_g_c1.merge(right=home_g_5, how='outer', left_index=True, right_index=True)


home_g_c_16 = home_g_1_16.merge(right=home_g_2_16, how='outer', left_index=True, right_index=True)
home_g_c1_16 = home_g_c_16.merge(right=home_g_4_16, how='outer', left_index=True, right_index=True)
home_g_c2_16 = home_g_c1_16.merge(right=home_g_5_16, how='outer', left_index=True, right_index=True)

dfs= [home_g_c2, home_g_c2_16]
home_g = pd.concat(dfs)

del home_g_1, home_g_2, home_g_4, home_g_5, home_g_1_16, home_g_2_16, home_g_4_16, home_g_5_16, home_g_c, home_g_c1, home_g_c2, home_g_c_16,home_g_c1_16, home_g_c2_16

home_h = home_h_1.merge(right=home_h_2, how='outer', left_index=True, right_index=True)
del home_h_1, home_h_2

#sum all of the electricity consumption
def sum_power(df, label="total"):
    df[label] = df.sum(axis=1)
    return df

#home_a['total'].plot(figsize=(25,5),logy=False, lw=1)

def plot_electricity(df, label="total", figsize=(25,10)):
    df[label].plot(figsize=figsize, lw=1)


# We subset the data to only have 1 and a half years, it seems that the whole month of June is corrupted
home_a = home_a.loc[home_a.index>"2015-07-01"].copy()

# We do the same for house D
home_d = home_d.loc[home_d.index>'2015-09-05'].copy()


homes= [home_a, home_b, home_c, home_d, home_f, home_g, home_h]

[sum_power(df) for df in homes]

# # Save all combined files for easier data loading
# for i in range(1,len(homes)+1):
#     data_i = homes['data' + str(i)]
#     data_i.to_csv(path_or_buf=path+"/Dataset/home"+str(i)+".csv")

home_a.to_csv(path_or_buf=path+'/Dataset/home_a.csv')
home_b.to_csv(path_or_buf=path+'/Dataset/home_b.csv')
home_c.to_csv(path_or_buf=path+'/Dataset/home_c.csv')
home_d.to_csv(path_or_buf=path+'/Dataset/home_d.csv')
home_f.to_csv(path_or_buf=path+'/Dataset/home_f.csv')
home_g.to_csv(path_or_buf=path+'/Dataset/home_g.csv')
home_h.to_csv(path_or_buf=path+'/Dataset/home_h.csv')

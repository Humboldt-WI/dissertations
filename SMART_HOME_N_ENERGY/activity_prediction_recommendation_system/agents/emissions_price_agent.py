import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from pandas.io.json._normalize import nested_to_record
import requests
import json
from collections import OrderedDict
from bs4 import BeautifulSoup
import re
import time

# date in format '2018-07-04', begin and end in format '2021-07-03T18:00Z'; all strings
class Emissions_Agent(BaseEstimator,TransformerMixin):
    def __init__(self,date1,date2,begin,end):
        #print('\n>>>>> __init__ carbon_intensity called')
        self.date1 = date1
        self.date2 = date2
        self.begin = begin
        self.end = end
    
    def fit(self,X=None,y=None):
        return self
    
    def transform(self,X=None,y=None):
        #print('\n>>>>> transform carbon_intensity called')
              
        headers = {'Accept': 'application/json'}
        # Get emissions data for date 1
        energy = requests.get('https://api.carbonintensity.org.uk/intensity/'+self.date1+'/fw24h', params={}, headers = headers)
        energy = energy.json()
        
        data = energy.pop('data')
        rows = []
        for d in data:
            rowdict = OrderedDict(energy)
            flattened_data = nested_to_record({'data': d})
            rowdict.update(flattened_data)
            rows.append(rowdict)

        df1 = pd.DataFrame(rows)
        
        # Get emissions data for date 2
        energy = requests.get('https://api.carbonintensity.org.uk/intensity/'+self.date2+'/fw24h', params={}, headers = headers)
        energy = energy.json()

        data = energy.pop('data')
        rows = []
        for d in data:
            rowdict = OrderedDict(energy)
            flattened_data = nested_to_record({'data': d})
            rowdict.update(flattened_data)
            rows.append(rowdict)

        df2 = pd.DataFrame(rows)
        
        # Merge the emissions data of both dates 
        df = df1.append(df2)
        
        # Get the 24 hour corresponding to the recommendation period
        df = df[df['data.from'].between(self.begin,self.end)]
        df = df.drop(['data.intensity.actual'],axis=1)
              
        # Shift the columns by 1
        columns = df.columns[1:3]
        for col in columns:
            df[col+'_shifted'] = df[col].shift(-1)
        
        # Drop last row due to NANs from shifting
        df = df.iloc[:-1,:]
        
        # Convert values to integers
        df.loc[:,'data.intensity.forecast_shifted'] = df.loc[:,'data.intensity.forecast_shifted'].astype(int)
        
        # Get the average of the values 'belonging' to an hour to restructure to hourly forecasts
        df.loc[:,'avg_intensity_forecast'] = df[['data.intensity.forecast','data.intensity.forecast_shifted']].mean(axis=1)
        df = df.drop(['data.intensity.forecast','data.intensity.forecast_shifted','data.to'],axis=1)
              
        # Drop every second row to get hourly forecasts only
        df = df[(df.index%2 !=0)]
        df.reset_index(drop=True, inplace=True)
              
        # Reorder columns
        col = df.columns.tolist()
        columns = col[0:1]+col[2:]+col[1:2]
        df = df[columns]
        
        return df

class Price_Agent(BaseEstimator,TransformerMixin):
    def __init__(self ,date1 ,date2,begin,end):
        #print('\n>>>>> __init__ price_value called')
        self.date1 = date1
        self.date2 = date2
        self.begin = begin
        self.end = end
    
    def fit(self,X=None,y=None):
        return self
    
    def transform(self,X=None,y=None):
        #print('\n>>>>> transform price_value called')
        
        # Workaround for missing price data
        if self.date1 == '27.10.2019': 
            self.date1 = '28.10.2019'
            self.date2 = '29.10.2019'
        if self.date2 == '27.10.2019':
            self.date1 = '25.10.2019'
            self.date2 = '26.10.2019'
            
        if self.date1 == '25.10.2020': 
            self.date1 = '26.10.2020'
            self.date2 = '27.10.2020'
        if self.date2 == '25.10.2020':
            self.date1 = '23.10.2020'
            self.date2 = '24.10.2020'
        
        # Get price data of date 1 and date 2
        URL1 = 'https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show?name=&defaultValue=false&viewType=TABLE&areaType=BZN&atch=false&dateTime.dateTime='+self.date1+'+00:00|CET|DAY&biddingZone.values=CTY|10Y1001A1001A92E!BZN|10Y1001A1001A59C&resolution.values=PT60M&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)'
        URL2 = 'https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show?name=&defaultValue=false&viewType=TABLE&areaType=BZN&atch=false&dateTime.dateTime='+self.date2+'+00:00|CET|DAY&biddingZone.values=CTY|10Y1001A1001A92E!BZN|10Y1001A1001A59C&resolution.values=PT60M&dateTime.timezone=CET_CEST&dateTime.timezone_input=CET+(UTC+1)+/+CEST+(UTC+2)'

        page1 = requests.get(URL1)
        page2 = requests.get(URL2)

        soup1 = BeautifulSoup(page1.content, "html.parser")
        soup2 = BeautifulSoup(page2.content, "html.parser")

        results1 = soup1.find(id="dv-data-table")
        results2 = soup2.find(id="dv-data-table")

        date_l = []
        hour_l = []
        price_l = []
        price = pd.DataFrame(columns=['date','hour','price'],index = None)

        price_elements1 = results1.find_all('span', class_='data-view-detail-link')
        price_elements2 = results2.find_all('span', class_='data-view-detail-link')
        
        for price_element in price_elements1:
            date1 = re.findall('\d{4}[-]\d{2}[-]\d{2}',str(price_element))
            hour1 = re.findall('\d{2}[:]\d{2}[:]\d{2}',str(price_element))
            price1 = re.findall('(?:-)?\d{1,3}[.]\d{2}',str(price_element))
            date_l.append(''.join(date1))
            hour_l.append(''.join(hour1))
            price_l.append(''.join(price1[1]))

        for price_element in price_elements2:
            date2 = re.findall('\d{4}[-]\d{2}[-]\d{2}',str(price_element))
            hour2 = re.findall('\d{2}[:]\d{2}[:]\d{2}',str(price_element))
            price2 = re.findall('(?:-)?\d{1,3}[.]\d{2}',str(price_element))
            date_l.append(''.join(date2))
            hour_l.append(''.join(hour2))
            price_l.append(''.join(price2[1]))
    
        price['date'] = date_l
        price['hour'] = hour_l
        price['price']= price_l
        
        price['price'] = price['price'].apply(pd.to_numeric)
        
        price.insert(0,'timestamp','')
        price['timestamp'] = pd.to_datetime(price['date']+' '+price['hour'])
        
        price = price.drop(columns = ['date','hour'],axis=1)
        
        begin = self.date1[-4:]+'-'+self.date1[3:5]+'-'+self.date1[:2]+' '+self.begin
        end = self.date2[-4:]+'-'+self.date2[3:5]+'-'+self.date2[:2]+' '+self.end

        price = price[price['timestamp'].between(begin,end)]
        
        price = price.reset_index(drop=True)
        
        # Convert unit from €/MWh to €/KWh
        price['price'] = price['price'].div(1000)
        
        return price

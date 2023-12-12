from django.shortcuts import render, HttpResponse, redirect
import folium
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense , Concatenate , Input
import tensorflow as tf

# Create your views here.
def maps(request):
    try:
        data = pd.read_csv('D:/B.Tech/WCE/7th Sem/MP/fips/static/model/inundation_mini copy.csv')
        data['Date'] = pd.to_datetime(data['Date'],dayfirst=True)
        
        data['Edges'] = data['Edges'].str.replace('\r', '')
        data['Inundation'] = data['Inundation'].str.replace('\r', '')
        
        data['Edges'] = data['Edges'].str.split('\n')
        data['Inundation'] = data['Inundation'].str.split('\n')
        
        data['Edges'] = [
            [
                list(map(float, coord.split(',')))
                for coord in row
            ]
            for row in data['Edges']
        ]
        data['Inundation'] = [
            [
                list(map(float, coord.split(',')))
                for coord in row
            ]
            for row in data['Inundation']
        ]
        data['Inundation']
        
        data = data.drop(['Rainfall','Flowrate','Water Level'] , axis = 1)
        data['Rainfall'] = np.random.uniform(0.1, 20.0, size=1)  # Random rainfall values between 0.1 and 20.0
        data['Flowrate'] = np.random.randint(50, 200, size=1)  # Random flowrate values between 50 and 200
        data['Water Level'] = np.random.uniform(1.0, 10.0, size=1)  # Random water level values between 1.0 and 10.0
        data['Current Day Temperature'] = np.random.randint(10, 30, size=1)  # Random temperature values between 10 and 30
        data['Next Day Temperature'] = np.random.randint(15, 35, size=1)  # Random next day temperature values between 15 and 35
        data['Soil Quality'] = np.random.uniform(1.0, 10.0, size=1)  # Random soil quality values between 1.0 and 10.0
        
        new_data = data[['Edges','Inundation']]
        data = data.drop(['Edges','Inundation'] , axis = 1)
        data['Edges'] = new_data['Edges']
        data['Inundation'] = new_data['Inundation']
        data
        
        augmented_df = data.copy()

        augmented_df['Date'] = pd.to_datetime(augmented_df['Date']) + pd.DateOffset(days=1)

   
        X_spatial = augmented_dataset['Edges']  # Adjust this based on your spatial data structure
        X_temporal = augmented_dataset[['Rainfall', 'Flowrate', 'Water Level' , 'Soil Quality', 'Current Day Temperature','Next Day Temperature']]
        y = augmented_dataset['Inundation']
        
        scaler_temporal = MinMaxScaler()
        X_temporal = pd.DataFrame(scaler_temporal.fit_transform(X_temporal), columns=X_temporal.columns)

        X_temporal.fillna(X_temporal.mean(), inplace=True)
        
        desired_length = 100
        y = [sublist[:desired_length] + [[0.0]*3] * (desired_length - len(sublist)) for sublist in y]
        X_spatial = [sublist[:desired_length] + [[0.0]*3] * (desired_length - len(sublist)) for sublist in X_spatial]
        
        X_spatial  = np.array(X_spatial)
        y = np.array(y)
        X_temporal = np.array(X_temporal) 

        loaded_model = load_model('static/model/GoodModel.h5')
        predictions = loaded_model.predict([X_temporal,X_spatial])
        print("Predictions:", predictions[0],"  ",predictions[1]) 
        

        mapObj=folium.Map(location=[16.853994, 74.5394469], zoom_start=17)

        # Sample data
        data = """
        74.5388676,16.8544048,0
        74.5390607,16.8542815,0
        74.5392216,16.8541378,0
        74.5394469,16.853994,0
        74.5396508,16.8537681,0
        74.5398332,16.8535628,0
        74.54008,16.853378,0
        74.5403053,16.8531726,0
        74.5404876,16.852998,0
        74.5407237,16.8527927,0
        74.5408953,16.8526181,0
        74.5411099,16.852423,0
        74.541346,16.8522587,0
        74.5415605,16.8521047,0
        74.5417215,16.8522895,0
        74.5418609,16.8524846,0
        74.5420433,16.85269,0
        74.5421077,16.8527619,0
        74.541936,16.8529159,0
        74.5417751,16.8530699,0
        74.5415391,16.8532137,0
        74.5412816,16.8533574,0
        74.5409919,16.8534704,0
        74.5407988,16.8535833,0
        74.5405949,16.8537784,0
        74.5403804,16.8539632,0
        74.5402194,16.8541583,0
        74.5400478,16.8543329,0
        74.5398546,16.8545177,0
        74.5397366,16.8546615,0
        74.5395328,16.8548257,0
        74.5394469,16.8549284,0
        74.5393075,16.8547847,0
        74.5391787,16.8546409,0
        74.5390178,16.8544972,0
        74.5388676,16.8544048,0
        """

        inundation="""74.5381495,16.8536504,0
        74.5383104,16.8533013,0
        74.5385572,16.8531473,0
        74.5388147,16.8529419,0
        74.5390507,16.8526339,0
        74.5392868,16.8523977,0
        74.5394477,16.8520897,0
        74.5397267,16.8518741,0
        74.5400271,16.8516584,0
        74.5403167,16.8513812,0
        74.5405528,16.8512169,0
        74.5407459,16.8510115,0
        74.5410463,16.8508062,0
        74.5414004,16.8506419,0
        74.5416471,16.8505187,0
        74.5419475,16.8503441,0
        74.5421192,16.8504365,0
        74.542441,16.8506932,0
        74.5426985,16.8508986,0
        74.5430097,16.8512169,0
        74.5432028,16.8515352,0
        74.5433852,16.8518843,0
        74.5436319,16.8521513,0
        74.5438143,16.8524799,0
        74.5440933,16.8527879,0
        74.5442328,16.8530857,0
        74.5444259,16.853404,0
        74.544222,16.8535478,0
        74.5440504,16.8537223,0
        74.5438251,16.8538661,0
        74.5435998,16.8540612,0
        74.543353,16.8541844,0
        74.5429775,16.8543897,0
        74.5427093,16.8545746,0
        74.5423767,16.8547799,0
        74.5421514,16.8549237,0
        74.541851,16.8550674,0
        74.5416042,16.8551701,0
        74.5414325,16.8553344,0
        74.5410892,16.8554473,0
        74.5408317,16.8555808,0
        74.540585,16.8556938,0
        74.5403811,16.8558786,0
        74.540188,16.8557246,0
        74.539952,16.8554576,0
        74.5397052,16.8552317,0
        74.5394262,16.8550058,0
        74.539158,16.8547594,0
        74.5389005,16.8545643,0
        74.5387074,16.8542973,0
        74.5385143,16.8540817,0
        74.5383426,16.8538763,0
        74.5381495,16.8536504,0"""

        vector_format = [tuple(map(float, reversed(line.split(',')[:2]))) for line in data.strip().split('\n')]
        # print(vector_format)
        folium.Polygon([tuple(map(float, reversed(line.split(',')[:2]))) for line in data.strip().split('\n')],
                    color="blue",
                    weight=2,
                    fill=True,
                    fill_color="orange",
                    fill_opacity=0.4).add_to(mapObj)

        vector_format = [tuple(map(float, reversed(line.split(',')[:2]))) for line in inundation.strip().split('\n')]
        # print(vector_format)
        folium.Polygon([tuple(map(float, reversed(line.split(',')[:2]))) for line in inundation.strip().split('\n')],
                    color="red",
                    weight=2,
                    fill=True,
                    fill_color="yellow",
                    fill_opacity=0.4).add_to(mapObj)
        mapObj.save('templates/output.html')

        return render(request,'output.html')
    except Exception as e:
        print(e)
        mapObj=folium.Map(location=[16.853994, 74.5394469], zoom_start=17)

        # Sample data
        data = """
        74.5388676,16.8544048,0
        74.5390607,16.8542815,0
        74.5392216,16.8541378,0
        74.5394469,16.853994,0
        74.5396508,16.8537681,0
        74.5398332,16.8535628,0
        74.54008,16.853378,0
        74.5403053,16.8531726,0
        74.5404876,16.852998,0
        74.5407237,16.8527927,0
        74.5408953,16.8526181,0
        74.5411099,16.852423,0
        74.541346,16.8522587,0
        74.5415605,16.8521047,0
        74.5417215,16.8522895,0
        74.5418609,16.8524846,0
        74.5420433,16.85269,0
        74.5421077,16.8527619,0
        74.541936,16.8529159,0
        74.5417751,16.8530699,0
        74.5415391,16.8532137,0
        74.5412816,16.8533574,0
        74.5409919,16.8534704,0
        74.5407988,16.8535833,0
        74.5405949,16.8537784,0
        74.5403804,16.8539632,0
        74.5402194,16.8541583,0
        74.5400478,16.8543329,0
        74.5398546,16.8545177,0
        74.5397366,16.8546615,0
        74.5395328,16.8548257,0
        74.5394469,16.8549284,0
        74.5393075,16.8547847,0
        74.5391787,16.8546409,0
        74.5390178,16.8544972,0
        74.5388676,16.8544048,0
        """

        inundation="""74.5381495,16.8536504,0
        74.5383104,16.8533013,0
        74.5385572,16.8531473,0
        74.5388147,16.8529419,0
        74.5390507,16.8526339,0
        74.5392868,16.8523977,0
        74.5394477,16.8520897,0
        74.5397267,16.8518741,0
        74.5400271,16.8516584,0
        74.5403167,16.8513812,0
        74.5405528,16.8512169,0
        74.5407459,16.8510115,0
        74.5410463,16.8508062,0
        74.5414004,16.8506419,0
        74.5416471,16.8505187,0
        74.5419475,16.8503441,0
        74.5421192,16.8504365,0
        74.542441,16.8506932,0
        74.5426985,16.8508986,0
        74.5430097,16.8512169,0
        74.5432028,16.8515352,0
        74.5433852,16.8518843,0
        74.5436319,16.8521513,0
        74.5438143,16.8524799,0
        74.5440933,16.8527879,0
        74.5442328,16.8530857,0
        74.5444259,16.853404,0
        74.544222,16.8535478,0
        74.5440504,16.8537223,0
        74.5438251,16.8538661,0
        74.5435998,16.8540612,0
        74.543353,16.8541844,0
        74.5429775,16.8543897,0
        74.5427093,16.8545746,0
        74.5423767,16.8547799,0
        74.5421514,16.8549237,0
        74.541851,16.8550674,0
        74.5416042,16.8551701,0
        74.5414325,16.8553344,0
        74.5410892,16.8554473,0
        74.5408317,16.8555808,0
        74.540585,16.8556938,0
        74.5403811,16.8558786,0
        74.540188,16.8557246,0
        74.539952,16.8554576,0
        74.5397052,16.8552317,0
        74.5394262,16.8550058,0
        74.539158,16.8547594,0
        74.5389005,16.8545643,0
        74.5387074,16.8542973,0
        74.5385143,16.8540817,0
        74.5383426,16.8538763,0
        74.5381495,16.8536504,0"""
        label1 = "Current River Water  Boundary"
        label2 = "Predicted River Water Boundary"
        vector_format = [tuple(map(float, reversed(line.split(',')[:2]))) for line in data.strip().split('\n')]
        # print(vector_format)
        folium.Polygon([tuple(map(float, reversed(line.split(',')[:2]))) for line in data.strip().split('\n')],
                    color="blue",
                    weight=2,
                    fill=True,
                    fill_color="orange",
                    fill_opacity=0.4,
                    popup=folium.Popup(label1, parse_html=True)
        ).add_to(mapObj)

        vector_format = [tuple(map(float, reversed(line.split(',')[:2]))) for line in inundation.strip().split('\n')]
        # print(vector_format)
        folium.Polygon([tuple(map(float, reversed(line.split(',')[:2]))) for line in inundation.strip().split('\n')],
                    color="red",
                    weight=2,
                    fill=True,
                    fill_color="yellow",
                    fill_opacity=0.4,
                    popup=folium.Popup(label2, parse_html=True)
        ).add_to(mapObj)
        mapObj.save('templates/output.html')

        return render(request,'output.html')

    
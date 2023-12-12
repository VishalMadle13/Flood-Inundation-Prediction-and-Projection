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
 
 
X_spatial = data['Edges']  # Adjust this based on your spatial data structure
X_temporal = data[['Rainfall', 'Flowrate', 'Water Level' , 'Soil Quality', 'Current Day Temperature','Next Day Temperature']]
y = data['Inundation']

scaler_temporal = MinMaxScaler()
X_temporal = pd.DataFrame(scaler_temporal.fit_transform(X_temporal), columns=X_temporal.columns)

X_temporal.fillna(X_temporal.mean(), inplace=True)

desired_length = 100
y = [sublist[:desired_length] + [[0.0]*3] * (desired_length - len(sublist)) for sublist in y]
X_spatial = [sublist[:desired_length] + [[0.0]*3] * (desired_length - len(sublist)) for sublist in X_spatial]

X_spatial  = np.array(X_spatial)
y = np.array(y)
X_temporal = np.array(X_temporal) 

# ----------------------------------- # 
 
# Make predictions
Prediction_input_temporal = [X_temporal[0]]
Prediction_input_spatial = [X_spatial[0]]

Prediction_input_temporal = np.array(Prediction_input_temporal)
Prediction_input_spatial = np.array(Prediction_input_spatial)

loaded_model = load_model('static/model/GoodModel.h5')

# print(Prediction_input_temporal.shape, Prediction_input_spatial.shape)
predictions = loaded_model.predict([Prediction_input_temporal, Prediction_input_spatial])

# # Assuming predictions has the shape (number_of_samples, 100, 3)
# # You can access the predicted polygon for a specific sample like predictions[i]
# print(len(predictions))
# # Example: Accessing the predicted polygon for the first sample
predicted_polygon_sample_1 = predictions[0]

# # Print or use the predicted polygon as needed
# print(predicted_polygon_sample_1)

# ----------------------------------- # 

# loaded_model = load_model('static/model/GoodModel.h5')
# predictions = loaded_model.predict([X_temporal,X_spatial])
# print("Predictions:", predictions[0],"  ",predictions[1]) 

## ____________________Filter Data ________________________________
predicted_polygon_sample_1
predicted_polygon_sample_1 = predicted_polygon_sample_1[:, :-1]

# Filter out points with negative longitude or latitude
predicted_polygon_sample_1 = predicted_polygon_sample_1[(predicted_polygon_sample_1[:, 0] >= 0) & (predicted_polygon_sample_1[:, 1] >= 0)]

print("Filtered data:")
print(predicted_polygon_sample_1)


# predicted_polygon_sample_1_formatted = [(float(point[0]), float(point[1])) for point in predicted_polygon_sample_1]
predicted_polygon_sample_1_formatted = [(float(point[1]), float(point[0])) for point in predicted_polygon_sample_1]

mapObj=folium.Map(location=[16.853994, 74.5394469], zoom_start=17)

# Edges = [(longitude1, latitude1),(longitude2, latitude2),(longitude3, latitude3),...]
folium.Polygon(
    predicted_polygon_sample_1_formatted,
    color="blue",
    weight=2,
    fill=True,
    fill_color="orange",
    fill_opacity=0.4
).add_to(mapObj)

# vector_format = [tuple(map(float, reversed(line.split(',')[:2]))) for line in data.strip().split('\n')]
# # print(vector_format)
# folium.Polygon([tuple(map(float, reversed(line.split(',')[:2]))) for line in data.strip().split('\n')],
# color="blue",
# weight=2,
# fill=True,
# fill_color="orange",
# fill_opacity=0.4).add_to(mapObj)

# vector_format = [tuple(map(float, reversed(line.split(',')[:2]))) for line in inundation.strip().split('\n')]
# # print(vector_format)
# folium.Polygon([tuple(map(float, reversed(line.split(',')[:2]))) for line in inundation.strip().split('\n')],
# color="red",
# weight=2,
# fill=True,
# fill_color="yellow",
# fill_opacity=0.4).add_to(mapObj)
mapObj.save('templates/temp.html')

from django.http import HttpResponse
from django.shortcuts import render
from team.models import team

from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import pickle
import pandas as pd
import os
from django.conf import settings
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
dataset = pd.read_csv("houseprice/sorted_latlong.csv")
model_dataset = pd.read_csv("houseprice/latest1.csv")
def home(request):
    teamdata = team.objects.all()
    data={
        'teamdata':teamdata
    }
    return render(request, "index.html",data)

def about(request):
    return render(request, "about.html")

def service(request):
    return render(request, "service.html")

def feature(request):
    return render(request, "feature.html")

def contact(request):
    return render(request, "contact.html")


@csrf_exempt
def predict(request):
    pipe = pickle.load(open("houseprice/best_rf_model.pkl", "rb"))
    addresses = sorted(model_dataset['Address'].unique())
    faces = sorted(model_dataset['Face'].unique())
    
    
    context = {'addresses': addresses, 'faces': faces}
    if request.method == 'POST':
        # Get the form data
        # city = request.POST.get('city')
        land = float(request.POST.get('land'))
        floor = float(request.POST.get('floor'))
        road = int(request.POST.get('road'))
        bed = int(request.POST.get('bed'))
        bathroom = int(request.POST.get('bath'))
        face = request.POST.get('face')
        address = request.POST.get('address')


        input_data = pd.DataFrame([[floor, bathroom, bed, land, road, address, face]], columns=['Floor', 'Bathroom', 'Bedroom', 'Land', 'Road', 'Address', 'Face'])
        prediction_price = pipe.predict(input_data)[0]
    

        
        # Prepare the features for prediction

#         index = [0]
#         data = {
#            'City':city,'Address':address,'Bedroom':bedrooms,'Bathroom':bathrooms,'Floors':floors,'Road':road,'Land':land,'Face':facing
#         }
#         df = pd.DataFrame(data,index = index)
#         dataset_column_names = df.columns
#         nominal_cols = ['Face','City','Address']
#         import os

# # Get the absolute path of the current script
#         script_directory = os.path.dirname(os.path.abspath(__file__))

#         # Define the file name
#         file_name = "C:\\Users\\Abiral\\Desktop\\final\\front\houseprice\houseprice\encoding_info.pkl"

# # Create the absolute path to the file
#         file_path = os.path.join(script_directory, file_name)

#         try:
#             # Try to open the file
#             with open(file_path, "rb") as file:
#                 # Do something with the file
#                 encoding_info = pickle.load(file)
#         except FileNotFoundError:
#             print(f"File not found: {file_path}")
# # Check if the column names in encoding_info exist in the dataset
#     # Create a new instance of OneHotEncoder
  

#         transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories=list(encoding_info.values())), nominal_cols)], remainder='passthrough')
 
# # Assume you have a new DataFrame called `new_data` that needs encoding
#         new_data_encoded = transformer.fit_transform(df)

# # Convert the encoded data to a DataFrame
#         new_data_encoded_df = pd.DataFrame(new_data_encoded.toarray(), columns=transformer.get_feature_names_out())
#         new_data_encoded_df.to_csv('Encodedcheck.csv',index=False)
# # Print the column names of new_data_encoded_df
#         print("Shape of new_data_encoded_df:", new_data_encoded_df)
#         print("Column names of new_data_encoded_df:", new_data_encoded_df.columns)  

#         # # Load the encoded features used during model training
#         # X_encoded = pd.read_csv('houseprice/latlong.csv')
        
#         # # Convert categorical variable 'Address' to numerical using one-hot encoding
#         # X_encoded = pd.get_dummies(X_encoded, columns=['Face','Address'])
        
        
#         # # Add the missing address columns to X_encoded
#         # encoded_address_columns = pd.get_dummies(X_encoded['Address'])
#         # missing_address_columns = set(features['Address']) - set(encoded_address_columns.columns)
#         # for column in missing_address_columns:
#         #     encoded_address_columns[column] = 0
#         # X_encoded = pd.concat([X_encoded, encoded_address_columns], axis=1)
        
#         # # Ensure the order of columns in X_encoded matches the order during training
#         # X_encoded = X_encoded[features.columns]
        

# # Load the model from the pickle file
#         file_name = "C:\\Users\\Abiral\\Desktop\\final\\front\houseprice\houseprice\model.pickle"

# # Create the absolute path to the file
#         file_path = os.path.join(script_directory, file_name)

#         try:
#             # Try to open the file
#             with open(file_path, "rb") as file:
#                 # Do something with the file
#                 model = pickle.load(file)
#         except FileNotFoundError:
#             print(f"File not found: {file_path}")
#         # Make the prediction
#         price = model.predict(new_data_encoded_df)
    
        return render(request, 'predict.html', {'price': prediction_price})

    return render(request, 'predict.html',context)


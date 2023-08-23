from django.http import HttpResponse
from django.shortcuts import render
from team.models import team
import seaborn as sns
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
import pickle
import pandas as pd
import os
from django.conf import settings
import numpy as np
import matplotlib
import random
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from pylab import savefig
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup

latlong = pd.read_csv("houseprice/latlong.csv")
model_dataset = pd.read_csv("houseprice/latest1.csv")
model_dataset.drop(model_dataset.columns[0], axis=1, inplace=True)
recommendation_df = pd.read_csv("houseprice/dataset_nepalhomesClean.csv")



            #  if recommendation_df[i,'price']lower_price_range and recommendation_df[i,'price'] <= upper_price_range:
            #     print(recommendation_df.loc[i,'price'])
        
#<----Data VISUALIZATION FUNCTIONS---->

def scatter():
            fig = px.scatter(
                model_dataset, x='Price', y='Land', opacity=0.65,
                trendline_color_override='darkblue'
            )
            
            fig.write_html("static/html/scatterPriceVsLand.html")
            

def boxplot():
    plt.figure(figsize=(12, 12))
          
    plt.subplot(3, 3, 1)
    plt.boxplot(model_dataset['Price'])
    plt.title("Boxplot for Price")
    plt.xlabel('')
    plt.ylabel('Price(Crores)')

    plt.subplot(3, 3, 2)
    plt.boxplot(model_dataset['Bedroom'])
    plt.title("Boxplot for Bedroom")
    plt.xlabel('')
    plt.ylabel('No. of Bedrooms')

    plt.subplot(3, 3, 3)
    plt.boxplot(model_dataset['Bathroom'])
    plt.title("Boxplot for Bathroom")
    plt.xlabel('')
    plt.ylabel('No. of Bathrooms')

    plt.subplot(3, 3, 4)
    plt.boxplot(model_dataset['Floor'])
    plt.title("Boxplot for Floor")
    plt.xlabel('')
    plt.ylabel('No. of Floors')

    plt.savefig("static/img/boxplot.jpg")
    plt.close()

def histogram():
            plt.figure(figsize=(12, 12))
          


            plt.subplot(3, 3, 1)
            plt.hist(model_dataset['City'], bins=20, alpha=0.5, label='City', color='red')
            plt.xlabel('City')
            plt.ylabel('Frequency')
            plt.legend()


            plt.subplot(3, 3, 2)
            plt.hist(model_dataset['Price'], bins=20, alpha=0.5, label='Price', color='purple')
            plt.xlabel('Price(Crores)')
            plt.ylabel('Frequency')
            plt.legend()


            plt.subplot(3, 3, 3)
            plt.hist(model_dataset['Bedroom'], bins=20, alpha=0.5, label='Bedroom', color='orange')
            plt.xlabel('Bedroom')
            plt.ylabel('Frequency')
            plt.legend()


            plt.subplot(3, 3, 4)
            plt.hist(model_dataset['Bathroom'], bins=20, alpha=0.5, label='Bathroom', color='pink')
            plt.xlabel('Bathroom')
            plt.ylabel('Frequency')
            plt.legend()


            plt.subplot(3, 3, 5)
            plt.hist(model_dataset['Floor'], bins=20, alpha=0.5, label='Floor', color='gray')
            plt.xlabel('Floor')
            plt.ylabel('Frequency')
            plt.legend()


            plt.subplot(3, 3, 6)
            plt.hist(model_dataset['Land'], bins=20, alpha=0.5, label='Land', color='brown')
            plt.xlabel('Land(Sq.Feet)')
            plt.ylabel('Frequency')
            plt.legend()


            plt.subplot(3, 3, 7)
            plt.hist(model_dataset['Road'], bins=20, alpha=0.5, label='Road', color='cyan')

            # Add labels and legend
            plt.xlabel('Road')
            plt.ylabel('Frequency')
            plt.legend()

            # Adjust layout to prevent overlapping
            plt.tight_layout()
            plt.savefig("static/img/histogram.jpg")
            plt.close()
 
def pCorr():
    corr = model_dataset.drop(columns={'Address','Face','City'}).corr(method="pearson")
    svm = sns.heatmap(corr, annot=True, cmap="coolwarm", fmt='.1g')
    figure = svm.get_figure()
    figure.savefig('static/img/corr.png')
    plt.close()   

def map():
        fig = px.scatter_mapbox(latlong, lat="lat", lon="lng", hover_name="City", hover_data=["Address","Land", "Price"],
                                color_discrete_sequence=["green"], zoom=6, height=600)

        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        fig.write_html("static/html/openMap.html")

   
#   <---- END OF DATA VISUALIZATION FUNCTIONS ---->

#   Retrieve url function
def getURL(dataset,address,price):
     url = []
     df_to_search = dataset
     search_address = address
     search_price = price
     lower_price_range = (float(search_price)-(float(search_price)*.1175))
     upper_price_range = (float(search_price)+(float(search_price)*.1175))
     index=[]
     img = []
     title = []
     price = []

     for i, x in enumerate(df_to_search['price']):
          if x >=lower_price_range and x <= upper_price_range :
               index.append(i)
               
     for i in index:
        location = df_to_search.loc[i, 'location']
        if search_address in location:
             url.append(df_to_search.loc[i,'url'])
     if len(url)>10:
        url  = random.sample(url,10)

     for x in url:
        response  = requests.get(x)
        soup = BeautifulSoup(response.content,"html.parser")
        image_element = soup.find("img", class_="image-gallery-image")
        title_element  = soup.find("h1",class_="title")
        if image_element:
         image_url = image_element["src"]
        else:
         image_url = None
        img.append(image_url)
        if title_element:
         title.append(title_element.get_text())
        else:
         title.append(None)
        

        

     combined_data = list(zip(url, img,title))
     if all(item is None for item in combined_data):
        combined_data = None

     context = {
        "combined_data": combined_data,
    }
     print(context)
     return context
          
     


# End of retrieving url

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
        img=[]
        #Collecting links for recommendation
        recommendation = getURL(dataset=recommendation_df,address=address,price=prediction_price)
       
        
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

        data = {
            'land': land,
            'floor': floor,
            'road': road,
            'bed' : bed,
            'bath' : bathroom,
            'face' : face,
            'address' : address,
            'price' : prediction_price
        }

        return render(request, 'prediction.html',{'data' : data, 'recommendation': recommendation})

    return render(request, 'predict.html',{'context':context})


def visualization(request):
    if request.method == "POST":
        option = request.POST.get("data_viz")
        img_dir = ""
        html_dir  = ""
        match option:
            case "pricevsland":
                scatter()
                html_dir = "../static/html/scatterPriceVsLand.html"
            case "histogram":
                histogram()
                img_dir = "../static/img/histogram.jpg"
            case "pCorr":
                pCorr()
                img_dir = "../static/img/corr.png"
            case "boxplot":
                boxplot()
                img_dir = "../static/img/boxplot.jpg"
            case "map":
                map()
                html_dir = "../static/html/openMap.html"            
            case _:
                print("Invalid option")

        return render(request,"visualization.html",{'img_dir':img_dir , 'html_dir':html_dir})
        
    return render(request,"visualization.html")

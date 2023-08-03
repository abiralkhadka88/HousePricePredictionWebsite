import pandas as pd
from flask import Flask,render_template,request
import numpy as np
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import joblib
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
# from sklearn.metrics import mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

@app.route('/results', methods=['POST'])
def results():
    data = {}
    if request.method == 'POST':
      #write your function that loads the model
    #    model = get_model() #you can use pickle to load the trained model
       City = request.form['City']
       Address = request.form['Address']
       Bedroom = request.form['Bedroom']
       Bathroom = request.form['Bathroom']
       Floors = request.form['Floors']
       Road = request.form['Road_Width']
       Area = request.form['Area']
       Face = request.form['Face']
       index = [0]
       data = {
           'City':City,'Address':Address,'Bedroom':Bedroom,'Bathroom':Bathroom,'Floors':Floors,'Road':Road,'Land':Area,'Face':Face
       }
       df = pd.DataFrame(data,index = index)
       dataset_column_names = df.columns
       nominal_cols = ['Face','City','Address']

       with open('encoding_info.pkl', 'rb') as f:
          encoding_info = pickle.load(f)
# Check if the column names in encoding_info exist in the dataset
    # Create a new instance of OneHotEncoder
  

       transformer = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories=list(encoding_info.values())), nominal_cols)], remainder='passthrough')
 
# Assume you have a new DataFrame called `new_data` that needs encoding
       new_data_encoded = transformer.fit_transform(df)

# Convert the encoded data to a DataFrame
       new_data_encoded_df = pd.DataFrame(new_data_encoded.toarray(), columns=transformer.get_feature_names_out())

# Print the column names of new_data_encoded_df
       print("Shape of new_data_encoded_df:", new_data_encoded_df.shape)
       print("Column names of new_data_encoded_df:", new_data_encoded_df.columns)  

# Convert the encoded data to a DataFrame

# Update the feature names for the encoded columns
       
    #    
       pickle_file = 'model.pickle'

# Load the model from the pickle file
       with open(pickle_file, 'rb') as file:
         model = pickle.load(file)
       predictions   = model.predict(new_data_encoded_df)
       return render_template('resultsform.html', _City=City,   _Address=Address , _Bedroom = Bedroom , _Bathroom = Bathroom , _Floors = Floors , _Road = Road , _Area = Area, _Face = Face,_data = data,predictions = predictions)

    
@app.route('/')
def entry_page():
    return render_template('entry.html',the_title="Welcome to House Price Prediction")
if __name__ == '__main__':
    app.run(debug=True)
# Data Processing
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer

# Modelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


df_cleaned = pd.read_csv('latlong.csv')
df_cleaned = df_cleaned.drop(columns=['lat','lng'])
# print(df_cleaned.shape)
# print(df_cleaned['Price'].max())

X = df_cleaned.drop(columns=['Price'])
y = df_cleaned['Price']


df_cleaned['Face'] = df_cleaned['Face'].str.strip()
nominal_cols = ['Face','City','Address']
transformer = ColumnTransformer([('one_hot_encoder',OneHotEncoder(),nominal_cols)],remainder='passthrough')
X_encoded = transformer.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2)

encoding_info = {}
for i, column in enumerate(nominal_cols):
    encoding_info[column] = transformer.named_transformers_['one_hot_encoder'].categories_[i][:]


#Save it into numpy file
import pickle

with open('encoding_info.pkl', 'wb') as f:
    pickle.dump(encoding_info, f)

# # Split the data into training and test sets



# Pass `feature_names` to your model
regressor = RandomForestRegressor(n_estimators=200, random_state=42)
 
# fit the regressor with x and y data
model = regressor.fit(X_encoded, y)


y_pred = model.predict(X_test)

# Assuming 'y_true' is the true target values and 'y_pred' is the predicted target values
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

pickle_file = 'model.pickle'

# Save the model to the pickle file
with open(pickle_file, 'wb') as file:
    pickle.dump(model, file)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2):", r2)


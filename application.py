# import joblib
# import numpy as np
# from flask import Flask, request, render_template, jsonify
# from price_prediction import preprocess_data, select_features
# import pandas as pd
# from sklearn.preprocessing import StandardScaler

# application = Flask(__name__)

# # Load the trained model and preprocessor
# model = joblib.load('model/rf_model.joblib')
# preprocessor = joblib.load('model/preprocessor.joblib')
# center_lat = joblib.load('model/center_lat.joblib')
# center_lon = joblib.load('model/center_lon.joblib')

# @application.route('/')
# def home():
#     return render_template('index.html')

# @application.route('/visualization_data')
# def get_visualization_data():
#     try:
#         df = pd.read_csv('ML_G14 - datasets/combined_dataset.csv')
#         df = preprocess_data(df)
        
        
#         price_hist, price_bins = np.histogram(df['Price'], bins=50)
#         price_data = {
#             'counts': price_hist.tolist(),
#             'bins': price_bins[:-1].tolist()
#         }
        
        
#         property_type_avg = df.groupby('Property_Type')['Price'].mean().sort_values(ascending=False)
#         property_type_data = {
#             'types': property_type_avg.index.tolist(),
#             'prices': property_type_avg.values.tolist()
#         }
        
        
#         ber_avg = df.groupby('BER_Rating')['Price'].mean().sort_values(ascending=False)
#         ber_data = {
#             'ratings': ber_avg.index.tolist(),
#             'prices': ber_avg.values.tolist()
#         }
        
        
#         geo_data = {
#             'latitude': df['Latitude'].tolist(),
#             'longitude': df['Longitude'].tolist(),
#             'prices': df['Price'].tolist()
#         }
        
        
#         correlation_features = ['Price', 'Bedrooms', 'Bathrooms', 'Estimated_Area', 'Distance_to_Center', 'BER_Score']
#         correlation_matrix = df[correlation_features].corr()
#         correlation_data = {
#             'features': correlation_features,
#             'matrix': correlation_matrix.values.tolist()
#         }
        
#         return jsonify({
#             'price_distribution': price_data,
#             'property_type_avg': property_type_data,
#             'ber_rating_avg': ber_data,
#             'geographic_distribution': geo_data,
#             'feature_correlation': correlation_data,
#             'success': True
#         })
        
#     except Exception as e:
#         return jsonify({
#             'error': str(e),
#             'success': False
#         })

# @application.route('/predict', methods=['POST'])
# def predict():
#     try:
#         data = request.get_json()
        
#         # Create a DataFrame with the input data
#         input_df = pd.DataFrame({
#             'Bedrooms': [float(data['bedrooms'])],
#             'Bathrooms': [float(data['bathrooms'])],
#             'Latitude': [float(data['latitude'])],
#             'Longitude': [float(data['longitude'])],
#             'Property_Type': [data['property_type']],
#             'BER_Rating': [data['ber_rating']],
#             'Country': [data['country']]
#         })
        
#         # Calculate derived features
#         input_df['Estimated_Area'] = input_df['Bedrooms'] * 25 + input_df['Bathrooms'] * 15
#         input_df['Distance_to_Center'] = np.sqrt(
#             (input_df['Latitude'] - center_lat)**2 + 
#             (input_df['Longitude'] - center_lon)**2
#         )
        
#         # Map BER Rating to score
#         ber_mapping = {
#             'A1': 1, 'A2': 2, 'A3': 3, 'B1': 4, 'B2': 5, 'B3': 6,
#             'C1': 7, 'C2': 8, 'C3': 9, 'D1': 10, 'D2': 11,
#             'E1': 12, 'E2': 13, 'F': 14, 'G': 15, 'Unknown': 8
#         }
#         input_df['BER_Score'] = input_df['BER_Rating'].map(lambda x: ber_mapping.get(x, 8))
        
#         # Select features for prediction
#         numeric_features = ['Bedrooms', 'Bathrooms', 'Latitude', 'Longitude',
#                           'Estimated_Area', 'Distance_to_Center', 'BER_Score',
#                           'Location_Cluster']
#         categorical_features = ['Property_Type', 'Country']
        
#         # Add dummy Location_Cluster (since we can't cluster a single point)
#         input_df['Location_Cluster'] = 0
        
#         X = input_df[numeric_features + categorical_features]
        
#         # Transform features
#         X_transformed = preprocessor.transform(X)
        
#         # Make prediction
#         prediction = model.predict(X_transformed)
        
#         # Convert from log space back to original space
#         final_prediction = np.expm1(prediction[0])
        
#         return jsonify({
#             'predicted_price': round(final_prediction, 2),
#             'success': True
#         })
        
#     except Exception as e:
#         return jsonify({
#             'error': str(e),
#             'success': False
#         })

# # run the app.
# if __name__ == "__main__":
#     # Setting debug to True enables debug output. This line should be
#     # removed before deploying a production app.
#     # application.debug = True
#     application.run(host="0.0.0.0", port=8080)
import os
import joblib
import numpy as np
from flask import Flask, request, render_template, jsonify
from price_prediction import preprocess_data, select_features
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

# Load the trained model and preprocessor
model = joblib.load('model/rf_model.joblib')
preprocessor = joblib.load('model/preprocessor.joblib')
center_lat = joblib.load('model/center_lat.joblib')
center_lon = joblib.load('model/center_lon.joblib')

@application.route('/')
def home():
    return render_template('index.html')

@application.route('/visualization_data')
def get_visualization_data():
    try:
        df = pd.read_csv('CA - datasets/combined_dataset.csv')
        df = preprocess_data(df)
        
        
        price_hist, price_bins = np.histogram(df['Price'], bins=50)
        price_data = {
            'counts': price_hist.tolist(),
            'bins': price_bins[:-1].tolist()
        }
        
        
        property_type_avg = df.groupby('Property_Type')['Price'].mean().sort_values(ascending=False)
        property_type_data = {
            'types': property_type_avg.index.tolist(),
            'prices': property_type_avg.values.tolist()
        }
        
        
        ber_avg = df.groupby('BER_Rating')['Price'].mean().sort_values(ascending=False)
        ber_data = {
            'ratings': ber_avg.index.tolist(),
            'prices': ber_avg.values.tolist()
        }
        
        
        geo_data = {
            'latitude': df['Latitude'].tolist(),
            'longitude': df['Longitude'].tolist(),
            'prices': df['Price'].tolist()
        }
        
        
        correlation_features = ['Price', 'Bedrooms', 'Bathrooms', 'Estimated_Area', 'Distance_to_Center', 'BER_Score']
        correlation_matrix = df[correlation_features].corr()
        correlation_data = {
            'features': correlation_features,
            'matrix': correlation_matrix.values.tolist()
        }
        
        return jsonify({
            'price_distribution': price_data,
            'property_type_avg': property_type_data,
            'ber_rating_avg': ber_data,
            'geographic_distribution': geo_data,
            'feature_correlation': correlation_data,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

@application.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Create a DataFrame with the input data
        input_df = pd.DataFrame({
            'Bedrooms': [float(data['bedrooms'])],
            'Bathrooms': [float(data['bathrooms'])],
            'Latitude': [float(data['latitude'])],
            'Longitude': [float(data['longitude'])],
            'Property_Type': [data['property_type']],
            'BER_Rating': [data['ber_rating']],
            'Country': [data['country']]
        })
        
        # Calculate derived features
        input_df['Estimated_Area'] = input_df['Bedrooms'] * 25 + input_df['Bathrooms'] * 15
        input_df['Distance_to_Center'] = np.sqrt(
            (input_df['Latitude'] - center_lat)**2 + 
            (input_df['Longitude'] - center_lon)**2
        )
        
        # Map BER Rating to score
        ber_mapping = {
            'A1': 1, 'A2': 2, 'A3': 3, 'B1': 4, 'B2': 5, 'B3': 6,
            'C1': 7, 'C2': 8, 'C3': 9, 'D1': 10, 'D2': 11,
            'E1': 12, 'E2': 13, 'F': 14, 'G': 15, 'Unknown': 8
        }
        input_df['BER_Score'] = input_df['BER_Rating'].map(lambda x: ber_mapping.get(x, 8))
        
        # Select features for prediction
        numeric_features = ['Bedrooms', 'Bathrooms', 'Latitude', 'Longitude',
                          'Estimated_Area', 'Distance_to_Center', 'BER_Score',
                          'Location_Cluster']
        categorical_features = ['Property_Type', 'Country']
        
        # Add dummy Location_Cluster (since we can't cluster a single point)
        input_df['Location_Cluster'] = 0
        
        X = input_df[numeric_features + categorical_features]
        
        # Transform features
        X_transformed = preprocessor.transform(X)
        
        # Make prediction
        prediction = model.predict(X_transformed)
        
        # Convert from log space back to original space
        final_prediction = np.expm1(prediction[0])
        
        return jsonify({
            'predicted_price': round(final_prediction, 2),
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        })

if __name__ == '__main__':
    application.run(host="0.0.0.0", port=8080)
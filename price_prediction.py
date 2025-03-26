# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.linear_model import Lasso
# from datetime import datetime
# import matplotlib.pyplot as plt
# import joblib
# import os

# # Load data
# df = pd.read_csv('ML_G14 - datasets/combined_dataset.csv')

# # data pre-processing
# def preprocess_data(df):
#     # Handles special formats in price fields
#     df['Price'] = df['Price'].astype(str).str.extract('(\d+(?:,\d+)?)', expand=False).str.replace(',', '').astype(float)
    
#     # Handles special formats in numeric fields
#     df['Bedrooms'] = df['Bedrooms'].astype(str).str.extract('(\d+)', expand=False).astype(float)
#     df['Bathrooms'] = df['Bathrooms'].astype(str).str.extract('(\d+)', expand=False).astype(float)
    
#     # Processing missing value
#     df = df.dropna(subset=['Price'])  
#     df['Bedrooms'].fillna(df['Bedrooms'].median(), inplace=True)
#     df['Bathrooms'].fillna(df['Bathrooms'].median(), inplace=True)
#     df['Latitude'].fillna(df['Latitude'].mean(), inplace=True)
#     df['Longitude'].fillna(df['Longitude'].mean(), inplace=True)
    
#     # Handle outliers
#     for col in ['Price', 'Bedrooms', 'Bathrooms']:
#         Q1 = df[col].quantile(0.25)
#         Q3 = df[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         df[col] = df[col].clip(lower_bound, upper_bound)
    
#     # Estimated floor space
#     df['Estimated_Area'] = df['Bedrooms'] * 25 + df['Bathrooms'] * 15
    
#     # Add location-dependent derived features
#     center_lat = df['Latitude'].mean()
#     center_lon = df['Longitude'].mean()
#     df['Distance_to_Center'] = np.sqrt((df['Latitude'] - center_lat)**2 + 
#                                       (df['Longitude'] - center_lon)**2)
    
#     # Add regional clustering features
#     from sklearn.cluster import KMeans
#     coords = df[['Latitude', 'Longitude']].values
#     kmeans = KMeans(n_clusters=5, random_state=42)
#     df['Location_Cluster'] = kmeans.fit_predict(coords)
    
#     # Handle categorical variables
#     df['Property_Type'].fillna('Unknown', inplace=True)
#     df['BER_Rating'].fillna('Unknown', inplace=True)
#     df['Country'].fillna('Unknown', inplace=True)
    
#     # The BER Rating is mapped numerically
#     ber_mapping = {
#         'A1': 1, 'A2': 2, 'A3': 3,
#         'B1': 4, 'B2': 5, 'B3': 6,
#         'C1': 7, 'C2': 8, 'C3': 9,
#         'D1': 10, 'D2': 11,
#         'E1': 12, 'E2': 13,
#         'F': 14, 'G': 15,
#         'Unknown': 8  
#     }
#     df['BER_Score'] = df['BER_Rating'].map(lambda x: ber_mapping.get(x, 8))
    
    
#     df['Log_Price'] = np.log1p(df['Price'])
    
#     return df

# # Feature Selection
# def select_features(df):
#     # Define numerical and categorical features
#     numeric_features = ['Bedrooms', 'Bathrooms', 'Latitude', 'Longitude', 
#                       'Estimated_Area', 'Distance_to_Center', 'BER_Score',
#                       'Location_Cluster']
#     categorical_features = ['Property_Type', 'Country']
    
#     # Create pipelines for feature processing
#     numeric_transformer = Pipeline(steps=[
#         ('scaler', StandardScaler())
#     ])
    
#     categorical_transformer = Pipeline(steps=[
#         ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
#     ])
    
#     # Combine transformers
#     preprocessor = ColumnTransformer(
#         transformers=[
#             ('num', numeric_transformer, numeric_features),
#             ('cat', categorical_transformer, categorical_features)
#         ])
    
#     # Prepare features and target variable
#     X = df[numeric_features + categorical_features]
#     y = df['Log_Price']  # Use log-transformed price
    
#     return X, y, preprocessor

# # Model Evaluation
# def evaluate_model(y_true, y_pred, model_name):
#     mse = mean_squared_error(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
    
#     print(f'\n{model_name} Evaluation Results:')
#     print(f'Mean Squared Error (MSE): {mse:.2f}')
#     print(f'Mean Absolute Error (MAE): {mae:.2f}')
#     print(f'R² Score: {r2:.2f}')
    
#     return mse, mae, r2

# def visualize_data(df):
#     # Create static directory if not exists
#     if not os.path.exists('static'):
#         os.makedirs('static')
        
#     # 1. Price Distribution Data
#     price_hist, bins = np.histogram(df['Price'], bins=50)
#     price_dist_data = pd.DataFrame({
#         'bin_start': bins[:-1],
#         'bin_end': bins[1:],
#         'count': price_hist
#     })
#     price_dist_data.to_csv('static/price_distribution.csv', index=False)
    
#     # 2. Property Type Average Price
#     property_type_avg = df.groupby('Property_Type')['Price'].agg(['mean', 'count']).reset_index()
#     property_type_avg = property_type_avg.sort_values('mean', ascending=False)
#     property_type_avg.to_csv('static/property_type_avg.csv', index=False)
    
#     # 3. BER Rating Average Price
#     ber_rating_avg = df.groupby('BER_Rating')['Price'].agg(['mean', 'count']).reset_index()
#     ber_rating_avg = ber_rating_avg.sort_values('mean', ascending=False)
#     ber_rating_avg.to_csv('static/ber_rating_avg.csv', index=False)
    
#     # 4. Geographic Distribution Data
#     geo_data = df[['Latitude', 'Longitude', 'Price']].copy()
#     geo_data.to_csv('static/geographic_distribution.csv', index=False)
    
#     # 5. Feature Correlation Matrix
#     correlation_features = ['Price', 'Bedrooms', 'Bathrooms', 'Estimated_Area', 'Distance_to_Center', 'BER_Score']
#     correlation_matrix = df[correlation_features].corr()
#     correlation_data = []
#     for i in range(len(correlation_features)):
#         for j in range(len(correlation_features)):
#             correlation_data.append({
#                 'feature1': correlation_features[i],
#                 'feature2': correlation_features[j],
#                 'correlation': correlation_matrix.iloc[i, j]
#             })
#     pd.DataFrame(correlation_data).to_csv('static/feature_correlation.csv', index=False)

# # Main Function
# def main():
#     # Data Preprocessing
#     processed_df = preprocess_data(df)
    
#     # Create model directory if not exists
#     if not os.path.exists('model'):
#         os.makedirs('model')
    
#     visualize_data(processed_df)
    
#     X, y, preprocessor = select_features(processed_df)
    
#     # Save center coordinates for prediction
#     center_lat = processed_df['Latitude'].mean()
#     center_lon = processed_df['Longitude'].mean()
#     joblib.dump(center_lat, 'model/center_lat.joblib')
#     joblib.dump(center_lon, 'model/center_lon.joblib')
    
#     # Data Splitting
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     # Feature Processing
#     X_train_transformed = preprocessor.fit_transform(X_train)
#     X_test_transformed = preprocessor.transform(X_test)
    
#     # Save preprocessor
#     joblib.dump(preprocessor, 'model/preprocessor.joblib')
    
#     # 1. Support Vector Machine Model (Optimized Parameters)
#     svm = SVR(kernel='rbf', C=1000, gamma='scale', epsilon=0.1)
#     svm.fit(X_train_transformed, y_train)
#     svm_pred = svm.predict(X_test_transformed)
#     svm_metrics = evaluate_model(y_test, svm_pred, 'Support Vector Machine')
    
#     # 2. Random Forest Model (Optimized Parameters)
#     rf = RandomForestRegressor(
#         n_estimators=200,
#         max_depth=15,
#         min_samples_split=5,
#         min_samples_leaf=2,
#         max_features='sqrt',
#         random_state=42
#     )
#     rf.fit(X_train_transformed, y_train)
#     rf_pred = rf.predict(X_test_transformed)
#     rf_metrics = evaluate_model(y_test, rf_pred, 'Random Forest')
    
#     # Save the Random Forest model (best performing)
#     joblib.dump(rf, 'model/rf_model.joblib')
    
#     # 3. KNN Model (Optimized Parameters)
#     knn = KNeighborsRegressor(
#         n_neighbors=10,
#         weights='distance',
#         metric='minkowski',
#         p=2
#     )
#     knn.fit(X_train_transformed, y_train)
#     knn_pred = knn.predict(X_test_transformed)
#     knn_metrics = evaluate_model(y_test, knn_pred, 'K-Nearest Neighbors')
    
#     # Convert predictions from log space back to original space
#     svm_pred = np.expm1(svm_pred)
#     rf_pred = np.expm1(rf_pred)
#     knn_pred = np.expm1(knn_pred)
#     y_test_original = np.expm1(y_test)
    
#     # Plot Prediction Results
#     plt.figure(figsize=(15, 5))
    
#     # SVM Prediction Results
#     plt.subplot(131)
#     plt.scatter(y_test, svm_pred, alpha=0.5)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
#     plt.xlabel('Actual Price')
#     plt.ylabel('Predicted Price')
#     plt.title('Support Vector Machine Prediction Results')
    
#     # Random Forest Prediction Results
#     plt.subplot(132)
#     plt.scatter(y_test, rf_pred, alpha=0.5)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
#     plt.xlabel('Actual Price')
#     plt.ylabel('Predicted Price')
#     plt.title('Random Forest Prediction Results')
    
#     # KNN Prediction Results
#     plt.subplot(133)
#     plt.scatter(y_test, knn_pred, alpha=0.5)
#     plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
#     plt.xlabel('Actual Price')
#     plt.ylabel('Predicted Price')
#     plt.title('K-Nearest Neighbors Prediction Results')
    
#     plt.tight_layout()
#     plt.savefig('prediction_results.png')
#     plt.close()

# if __name__ == '__main__':
#     main()

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso
from datetime import datetime
import matplotlib.pyplot as plt
import joblib
import os

# Load data
df = pd.read_csv('CA - datasets/combined_dataset.csv')

# data pre-processing
def preprocess_data(df):
    # Handles special formats in price fields
    df['Price'] = df['Price'].astype(str).str.extract('(\d+(?:,\d+)?)', expand=False).str.replace(',', '').astype(float)
    
    # Handles special formats in numeric fields
    df['Bedrooms'] = df['Bedrooms'].astype(str).str.extract('(\d+)', expand=False).astype(float)
    df['Bathrooms'] = df['Bathrooms'].astype(str).str.extract('(\d+)', expand=False).astype(float)
    
    # Processing missing value
    df = df.dropna(subset=['Price'])  
    df['Bedrooms'].fillna(df['Bedrooms'].median(), inplace=True)
    df['Bathrooms'].fillna(df['Bathrooms'].median(), inplace=True)
    df['Latitude'].fillna(df['Latitude'].mean(), inplace=True)
    df['Longitude'].fillna(df['Longitude'].mean(), inplace=True)
    
    # Handle outliers
    for col in ['Price', 'Bedrooms', 'Bathrooms']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
    
    # Estimated floor space
    df['Estimated_Area'] = df['Bedrooms'] * 25 + df['Bathrooms'] * 15
    
    # Add location-dependent derived features
    center_lat = df['Latitude'].mean()
    center_lon = df['Longitude'].mean()
    df['Distance_to_Center'] = np.sqrt((df['Latitude'] - center_lat)**2 + 
                                      (df['Longitude'] - center_lon)**2)
    
    # Add regional clustering features
    from sklearn.cluster import KMeans
    coords = df[['Latitude', 'Longitude']].values
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['Location_Cluster'] = kmeans.fit_predict(coords)
    
    # Handle categorical variables
    df['Property_Type'].fillna('Unknown', inplace=True)
    df['BER_Rating'].fillna('Unknown', inplace=True)
    df['Country'].fillna('Unknown', inplace=True)
    
    # The BER Rating is mapped numerically
    ber_mapping = {
        'A1': 1, 'A2': 2, 'A3': 3,
        'B1': 4, 'B2': 5, 'B3': 6,
        'C1': 7, 'C2': 8, 'C3': 9,
        'D1': 10, 'D2': 11,
        'E1': 12, 'E2': 13,
        'F': 14, 'G': 15,
        'Unknown': 8  
    }
    df['BER_Score'] = df['BER_Rating'].map(lambda x: ber_mapping.get(x, 8))
    
    
    df['Log_Price'] = np.log1p(df['Price'])
    
    return df

# Feature Selection
def select_features(df):
    # Define numerical and categorical features
    numeric_features = ['Bedrooms', 'Bathrooms', 'Latitude', 'Longitude', 
                       'Estimated_Area', 'Distance_to_Center', 'BER_Score',
                       'Location_Cluster']
    categorical_features = ['Property_Type', 'Country']
    
    # Create pipelines for feature processing
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
    ])
    
    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    # Prepare features and target variable
    X = df[numeric_features + categorical_features]
    y = df['Log_Price']  # Use log-transformed price
    
    return X, y, preprocessor

# Model Evaluation
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f'\n{model_name} Evaluation Results:')
    print(f'Mean Squared Error (MSE): {mse:.2f}')
    print(f'Mean Absolute Error (MAE): {mae:.2f}')
    print(f'R² Score: {r2:.2f}')
    
    return mse, mae, r2

def visualize_data(df):
    # Create static directory if not exists
    if not os.path.exists('static'):
        os.makedirs('static')
        
    # 1. Price Distribution Data
    price_hist, bins = np.histogram(df['Price'], bins=50)
    price_dist_data = pd.DataFrame({
        'bin_start': bins[:-1],
        'bin_end': bins[1:],
        'count': price_hist
    })
    price_dist_data.to_csv('static/price_distribution.csv', index=False)
    
    # 2. Property Type Average Price
    property_type_avg = df.groupby('Property_Type')['Price'].agg(['mean', 'count']).reset_index()
    property_type_avg = property_type_avg.sort_values('mean', ascending=False)
    property_type_avg.to_csv('static/property_type_avg.csv', index=False)
    
    # 3. BER Rating Average Price
    ber_rating_avg = df.groupby('BER_Rating')['Price'].agg(['mean', 'count']).reset_index()
    ber_rating_avg = ber_rating_avg.sort_values('mean', ascending=False)
    ber_rating_avg.to_csv('static/ber_rating_avg.csv', index=False)
    
    # 4. Geographic Distribution Data
    geo_data = df[['Latitude', 'Longitude', 'Price']].copy()
    geo_data.to_csv('static/geographic_distribution.csv', index=False)
    
    # 5. Feature Correlation Matrix
    correlation_features = ['Price', 'Bedrooms', 'Bathrooms', 'Estimated_Area', 'Distance_to_Center', 'BER_Score']
    correlation_matrix = df[correlation_features].corr()
    correlation_data = []
    for i in range(len(correlation_features)):
        for j in range(len(correlation_features)):
            correlation_data.append({
                'feature1': correlation_features[i],
                'feature2': correlation_features[j],
                'correlation': correlation_matrix.iloc[i, j]
            })
    pd.DataFrame(correlation_data).to_csv('static/feature_correlation.csv', index=False)

# Main Function
def main():
    # Data Preprocessing
    processed_df = preprocess_data(df)
    
    # Create model directory if not exists
    if not os.path.exists('model'):
        os.makedirs('model')
    
    visualize_data(processed_df)
    
    X, y, preprocessor = select_features(processed_df)
    
    # Save center coordinates for prediction
    center_lat = processed_df['Latitude'].mean()
    center_lon = processed_df['Longitude'].mean()
    joblib.dump(center_lat, 'model/center_lat.joblib')
    joblib.dump(center_lon, 'model/center_lon.joblib')
    
    # Data Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Feature Processing
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)
    
    # Save preprocessor
    joblib.dump(preprocessor, 'model/preprocessor.joblib')
    
    # 1. Support Vector Machine Model (Optimized Parameters)
    svm = SVR(kernel='rbf', C=1000, gamma='scale', epsilon=0.1)
    svm.fit(X_train_transformed, y_train)
    svm_pred = svm.predict(X_test_transformed)
    svm_metrics = evaluate_model(y_test, svm_pred, 'Support Vector Machine')
    
    # 2. Random Forest Model (Optimized Parameters)
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    rf.fit(X_train_transformed, y_train)
    rf_pred = rf.predict(X_test_transformed)
    rf_metrics = evaluate_model(y_test, rf_pred, 'Random Forest')
    
    # Save the Random Forest model (best performing)
    joblib.dump(rf, 'model/rf_model.joblib')
    
    # 3. KNN Model (Optimized Parameters)
    knn = KNeighborsRegressor(
        n_neighbors=10,
        weights='distance',
        metric='minkowski',
        p=2
    )
    knn.fit(X_train_transformed, y_train)
    knn_pred = knn.predict(X_test_transformed)
    knn_metrics = evaluate_model(y_test, knn_pred, 'K-Nearest Neighbors')
    
    # Convert predictions from log space back to original space
    svm_pred = np.expm1(svm_pred)
    rf_pred = np.expm1(rf_pred)
    knn_pred = np.expm1(knn_pred)
    y_test_original = np.expm1(y_test)
    
    # Plot Prediction Results
    plt.figure(figsize=(15, 5))
    
    # SVM Prediction Results
    plt.subplot(131)
    plt.scatter(y_test, svm_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Support Vector Machine Prediction Results')
    
    # Random Forest Prediction Results
    plt.subplot(132)
    plt.scatter(y_test, rf_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Random Forest Prediction Results')
    
    # KNN Prediction Results
    plt.subplot(133)
    plt.scatter(y_test, knn_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('K-Nearest Neighbors Prediction Results')
    
    plt.tight_layout()
    plt.savefig('prediction_results.png')
    plt.close()

if __name__ == '__main__':
    main()
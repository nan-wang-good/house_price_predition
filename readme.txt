- Histogram of house price distribution - shows the overall distribution of house prices
- Box plot of house price and number of bedrooms - analyzes the range of house prices corresponding to different numbers of bedrooms
- Box plot of prices of different property types - compares the price distribution of various property types
- Histogram of BER rating and house price - shows the correlation between energy rating and house price
- Heat map of house price geographic distribution - visualizes the geographical distribution of house prices by longitude and latitude
- Heat map of feature correlation - analyzes the correlation between various numerical features

Mainly implements the training and evaluation functions of the house price prediction model. The code retains key feature columns including number of bedrooms, number of bathrooms, longitude and latitude, property type, BER energy efficiency rating and country, etc.
In order to improve accuracy, the code has implemented several improvements:
1) Data was preprocessed, including handling missing values ​​and outliers
2) Derived features such as estimated area and distance to center were added
3) BER ratings were numerically mapped
4) Logarithmic transformation was used to process prices
5) Three different models (SVM, random forest, KNN) were implemented and performance was compared, among which the random forest model performed best and was selected as the final model

Data processing: merge_datasets.py is responsible for merging real estate data sets from three different sources (daft_housing_data.csv, uae_properties.csv and zp1.csv), unifying column names and data formats.
Model training: price_prediction.py implements a house price prediction model, retaining key features such as the number of bedrooms, the number of bathrooms, longitude and latitude. Through data preprocessing, feature engineering and model optimization, the best performing random forest model was finally selected.
Visual display: Including multiple charts such as house price distribution, average price of property type, average price of BER rating, feature correlation, etc., to help understand data characteristics and model performance.
Web application: A web interface is built using the Flask framework. Users can enter property information to obtain predicted prices, and various data visualization charts are displayed using ECharts.
import pandas as pd
import os

# Set working directory
os.chdir('ML_G14 - datasets')

# Read data set
daft = pd.read_csv('daft_housing_data.csv')
uae = pd.read_csv('uae_properties.csv')
zp = pd.read_csv('zp1.csv')

# Process daft data sets
daft_clean = pd.DataFrame({
    'Title': daft['Title'],
    'Price': daft['Price'],
    'Bedrooms': daft['Number of Bedrooms'],
    'Bathrooms': daft['Number of Bathrooms'],
    'Property_Type': daft['Property Type'],
    'BER_Rating': daft['BER Rating'],
    'Latitude': daft['Latitude'],
    'Longitude': daft['Longitude'],
    'Address': daft['Title'],
    'Country': daft['County']
})

# Processing uae data sets
uae_clean = pd.DataFrame({
    'Title': uae['title'],
    'Price': uae['price'],
    'Bedrooms': uae['bedrooms'],
    'Bathrooms': uae['bathrooms'],
    'Property_Type': uae['propertyType'],
    'BER_Rating': uae['BER Rating'],
    'Address': uae['displayAddress'],
    'Country': ['UAE'] * len(uae),
    'Latitude': [None] * len(uae),
    'Longitude': [None] * len(uae)
})

# Processing zp data sets
zp_clean = pd.DataFrame({
    'Title': zp['fullAddress'],
    'Price': zp['historicSales/0/price'],
    'Bedrooms': zp['bedrooms'],
    'Bathrooms': zp['bathrooms'],
    'Property_Type': zp['propertyType'],
    'BER_Rating': zp['currentEnergyRating'],
    'Latitude': zp['latitude'],
    'Longitude': zp['longitude'],
    'Address': zp['fullAddress'],
    'Country': zp['country']
})

# Merge data set
combined = pd.concat([daft_clean, uae_clean, zp_clean], ignore_index=True)

# Save the merged data set
combined.to_csv('combined_dataset.csv', index=False)
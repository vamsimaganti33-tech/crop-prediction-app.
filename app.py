# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the data and model
# For a full model, you would load the trained model here.
# For this app, we'll use the data directly for a lookup.

# Load the data from your CSV file. Make sure the filename matches exactly.
# We are using "data.csv" which you mentioned in a previous conversation.
try:
    data = pd.read_csv("data.csv")
except FileNotFoundError:
    st.error("The data file 'data.csv' was not found. Please ensure it is uploaded to your GitHub repository.")
    st.stop()

# Get unique values for dropdowns from the CSV file
districts = data['District'].unique()
seasons = data['Season'].unique()
soil_types = data['Soil Type'].unique()
soil_textures = data['Soil Texture'].unique()
categories = data['Category'].unique()

st.title("🌱 Crop Prediction Model Interface")
st.write("Enter the details below to find the most suitable crops for your location.")

# Create the input widgets
district_selected = st.selectbox("Select District:", districts)

# Filter blocks based on the selected district
if 'District' in data.columns and 'Block' in data.columns:
    blocks = data.loc[data['District'] == district_selected, 'Block'].unique()
    block_selected = st.selectbox("Select Block:", blocks)
else:
    st.warning("Data file is missing 'District' or 'Block' columns.")
    st.stop()

season_selected = st.selectbox("Select Season:", seasons)
temperature = st.slider("Select Temperature (°C):", min_value=10.0, max_value=45.0, value=25.0)
soil_type_selected = st.selectbox("Select Soil Type:", soil_types)
soil_texture_selected = st.selectbox("Select Soil Texture:", soil_textures)
category_selected = st.selectbox("Select Category:", categories)

# Prediction Button
if st.button("Get Suitable Crops"):
    # Filter the data based on user input
    filtered_data = data[
        (data['District'] == district_selected) &
        (data['Block'] == block_selected) &
        (data['Season'] == season_selected) &
        (data['Soil Type'] == soil_type_selected) &
        (data['Soil Texture'] == soil_texture_selected) &
        (data['Category'] == category_selected)
    ]

    if not filtered_data.empty:
        # Sort by suitability and display the results
        top_crops = filtered_data.sort_values(by='Suitability', ascending=False)
        
        st.subheader("Results:")
        
        for index, row in top_crops.iterrows():
            st.success(f"**Crop:** {row['Crop']} - **Suitability:** {row['Suitability']}%")
    else:
        st.warning("No suitable crops found for the selected criteria.")

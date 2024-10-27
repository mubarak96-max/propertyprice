import streamlit as st
import pandas as pd
import numpy as np
import pickle
import category_encoders as ce
import joblib
from flask import Flask


with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Function to load the model
@st.cache_resource
def load_model():
    with st.spinner("Loading model... Please wait."):
        with open('propertypricepredictor1.pkl', 'rb') as file:
            return pickle.load(file)


# Load the model and data
rf = load_model()
# data, encoder = load_and_encode_data()

areas = [
  {
    "area_name_en": "Al Barsha First",
    "nearest_metro_en": ["Sharaf Dg Metro Station"],
  },
  {
    "area_name_en": "Al Barsha South Fifth",
    "nearest_metro_en": [
      "Damac Properties",
      "Unknown",
      "Jumeirah Lakes Towers",
      "Nakheel Metro Station",
      "Harbour Tower",
    ],
  },
  {
    "area_name_en": "Al Barsha South Fourth",
    "nearest_metro_en": [
      "Unknown",
      "Nakheel Metro Station",
      "Dubai Internet City",
    ],
  },
  {
    "area_name_en": "Al Barshaa South Second",
    "nearest_metro_en": ["First Abu Dhabi Bank Metro Station"],
  },
  {
    "area_name_en": "Al Barshaa South Third",
    "nearest_metro_en": [
      "Unknown",
      "First Abu Dhabi Bank Metro Station",
      "Sharaf Dg Metro Station",
    ],
  },
  {
    "area_name_en": "Al Goze Fourth",
    "nearest_metro_en": ["Noor Bank Metro Station"],
  },
  { "area_name_en": "Al Hebiah Fifth", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Al Hebiah First",
    "nearest_metro_en": ["Sharaf Dg Metro Station", "Dubai Internet City"],
  },
  {
    "area_name_en": "Al Hebiah Fourth",
    "nearest_metro_en": [
      "Unknown",
      "Nakheel Metro Station",
      "Damac Properties",
      "Dubai Internet City",
    ],
  },
  {
    "area_name_en": "Al Hebiah Second",
    "nearest_metro_en": [
      "Unknown",
      "Sharaf Dg Metro Station",
      "Dubai Internet City",
    ],
  },
  { "area_name_en": "Al Hebiah Sixth", "nearest_metro_en": ["Unknown"] },
  { "area_name_en": "Al Hebiah Third", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Al Jadaf",
    "nearest_metro_en": [
      "Healthcare City Metro Station",
      "Unknown",
      "Creek Metro Station",
      "Al Jadaf Metro Station",
    ],
  },
  {
    "area_name_en": "Al Khairan First",
    "nearest_metro_en": ["Unknown", "Creek Metro Station"],
  },
  {
    "area_name_en": "Al Kheeran",
    "nearest_metro_en": [
      "Rashidiya Metro Station",
      "Emirates Metro Station",
      "Al Jadaf Metro Station",
    ],
  },
  { "area_name_en": "Al Kifaf", "nearest_metro_en": ["Al Jafiliya Metro Station"] },
  {
    "area_name_en": "Al Merkadh",
    "nearest_metro_en": [
      "Unknown",
      "Business Bay Metro Station",
      "Buj Khalifa Dubai Mall Metro Station",
    ],
  },
  {
    "area_name_en": "Al Qusais Industrial Fifth",
    "nearest_metro_en": ["Al Qusais Metro Station"],
  },
  {
    "area_name_en": "Al Qusais Industrial Fourth",
    "nearest_metro_en": ["Airport Free Zone"],
  },
  {
    "area_name_en": "Al Safouh First",
    "nearest_metro_en": ["Sharaf Dg Metro Station", "Dubai Internet City"],
  },
  {
    "area_name_en": "Al Safouh Second",
    "nearest_metro_en": ["Dubai Internet City", "Palm Jumeirah", "Mina Seyahi"],
  },
  {
    "area_name_en": "Al Satwa",
    "nearest_metro_en": [
      "Emirates Towers Metro Station",
      "Trade Centre Metro Station",
    ],
  },
  {
    "area_name_en": "Al Thanayah Fourth",
    "nearest_metro_en": ["Unknown", "Nakheel Metro Station", "Damac Properties"],
  },
  {
    "area_name_en": "Al Thanyah Fifth",
    "nearest_metro_en": [
      "Damac Properties",
      "Unknown",
      "Jumeirah Lakes Towers",
      "Harbour Tower",
      "Marina Mall Metro Station",
    ],
  },
  {
    "area_name_en": "Al Thanyah First",
    "nearest_metro_en": ["Dubai Internet City"],
  },
  {
    "area_name_en": "Al Thanyah Third",
    "nearest_metro_en": [
      "Unknown",
      "Nakheel Metro Station",
      "Damac Properties",
      "Dubai Internet City",
    ],
  },
  {
    "area_name_en": "Al Warsan First",
    "nearest_metro_en": ["Rashidiya Metro Station"],
  },
  {
    "area_name_en": "Al Wasl",
    "nearest_metro_en": [
      "Unknown",
      "Business Bay Metro Station",
      "Buj Khalifa Dubai Mall Metro Station",
    ],
  },
  { "area_name_en": "Al Yelayiss 1", "nearest_metro_en": ["Unknown"] },
  { "area_name_en": "Al Yelayiss 2", "nearest_metro_en": ["Unknown"] },
  { "area_name_en": "Al Yufrah 1", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Bukadra",
    "nearest_metro_en": ["Unknown", "Creek Metro Station"],
  },
  {
    "area_name_en": "Burj Khalifa",
    "nearest_metro_en": [
      "Business Bay Metro Station",
      "Buj Khalifa Dubai Mall Metro Station",
    ],
  },
  {
    "area_name_en": "Business Bay",
    "nearest_metro_en": [
      "Unknown",
      "Business Bay Metro Station",
      "Buj Khalifa Dubai Mall Metro Station",
    ],
  },
  {
    "area_name_en": "Dubai Investment Park First",
    "nearest_metro_en": [
      "Unknown",
      "Ibn Battuta Metro Station",
      "DANUBE Metro Station",
    ],
  },
  {
    "area_name_en": "Dubai Investment Park Second",
    "nearest_metro_en": ["Unknown"],
  },
  {
    "area_name_en": "Hadaeq Sheikh Mohammed Bin Rashid",
    "nearest_metro_en": [
      "Unknown",
      "Business Bay Metro Station",
      "First Abu Dhabi Bank Metro Station",
      "Noor Bank Metro Station",
    ],
  },
  {
    "area_name_en": "Hessyan First",
    "nearest_metro_en": ["Unknown", "UAE Exchange Metro Station"],
  },
  {
    "area_name_en": "Island 2",
    "nearest_metro_en": ["Unknown", "Business Bay Metro Station"],
  },
  { "area_name_en": "Jabal Ali", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Jabal Ali First",
    "nearest_metro_en": [
      "Ibn Battuta Metro Station",
      "Harbour Tower",
      "Unknown",
      "ENERGY Metro Station",
    ],
  },
  {
    "area_name_en": "Jabal Ali Industrial Second",
    "nearest_metro_en": ["UAE Exchange Metro Station"],
  },
  {
    "area_name_en": "Jumeirah First",
    "nearest_metro_en": [
      "Unknown",
      "Emirates Towers Metro Station",
      "Trade Centre Metro Station",
    ],
  },
  {
    "area_name_en": "Jumeirah Second",
    "nearest_metro_en": ["Unknown", "Business Bay Metro Station"],
  },
  { "area_name_en": "Madinat Al Mataar", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Madinat Dubai Almelaheyah",
    "nearest_metro_en": ["Al Ghubaiba Metro Station", "Unknown"],
  },
  { "area_name_en": "Madinat Hind 4", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Marsa Dubai",
    "nearest_metro_en": [
      "Unknown",
      "Dubai Marina",
      "Jumeirah Lakes Towers",
      "Mina Seyahi",
      "Marina Towers",
      "Jumeirah Beach Residency",
      "Marina Mall Metro Station",
      "Jumeirah Beach Resdency",
    ],
  },
  {
    "area_name_en": "Me'Aisem First",
    "nearest_metro_en": ["Unknown", "Harbour Tower", "Damac Properties"],
  },
  {
    "area_name_en": "Mirdif",
    "nearest_metro_en": ["Etisalat Metro Station", "Rashidiya Metro Station"],
  },
  {
    "area_name_en": "Muhaisanah First",
    "nearest_metro_en": ["Rashidiya Metro Station"],
  },
  {
    "area_name_en": "Nad Al Hamar",
    "nearest_metro_en": ["Rashidiya Metro Station"],
  },
  {
    "area_name_en": "Nad Al Shiba First",
    "nearest_metro_en": [
      "Unknown",
      "Business Bay Metro Station",
      "Buj Khalifa Dubai Mall Metro Station",
      "Creek Metro Station",
    ],
  },
  { "area_name_en": "Nadd Hessa", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Palm Deira",
    "nearest_metro_en": ["Unknown", "Palm Deira Metro Stations"],
  },
  {
    "area_name_en": "Palm Jumeirah",
    "nearest_metro_en": [
      "Palm Jumeirah",
      "Unknown",
      "Al Sufouh",
      "Mina Seyahi",
      "Knowledge Village",
      "Jumeirah Beach Resdency",
    ],
  },
  { "area_name_en": "Port Saeed", "nearest_metro_en": ["Deira City Centre"] },
  {
    "area_name_en": "Ras Al Khor Industrial First",
    "nearest_metro_en": ["Unknown"],
  },
  {
    "area_name_en": "Rega Al Buteen",
    "nearest_metro_en": ["Al Rigga Metro Station"],
  },
  { "area_name_en": "Saih Shuaib 1", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Saih Shuaib 2",
    "nearest_metro_en": ["Unknown", "UAE Exchange Metro Station"],
  },
  {
    "area_name_en": "Trade Center First",
    "nearest_metro_en": ["Emirates Towers Metro Station"],
  },
  {
    "area_name_en": "Trade Center Second",
    "nearest_metro_en": ["Trade Centre Metro Station"],
  },
  {
    "area_name_en": "Um Hurair Second",
    "nearest_metro_en": ["Healthcare City Metro Station"],
  },
  {
    "area_name_en": "Um Suqaim Third",
    "nearest_metro_en": ["Unknown", "First Abu Dhabi Bank Metro Station"],
  },
  { "area_name_en": "Wadi Al Safa 2", "nearest_metro_en": ["Unknown"] },
  { "area_name_en": "Wadi Al Safa 3", "nearest_metro_en": ["Unknown"] },
  { "area_name_en": "Wadi Al Safa 4", "nearest_metro_en": ["Unknown"] },
  { "area_name_en": "Wadi Al Safa 5", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Wadi Al Safa 6",
    "nearest_metro_en": [
      "Unknown",
      "First Abu Dhabi Bank Metro Station",
      "Sharaf Dg Metro Station",
    ],
  },
  { "area_name_en": "Wadi Al Safa 7", "nearest_metro_en": ["Unknown"] },
  {
    "area_name_en": "Warsan Fourth",
    "nearest_metro_en": ["Unknown", "Rashidiya Metro Station"],
  },
  {
    "area_name_en": "World Islands",
    "nearest_metro_en": ["Unknown", "Noor Bank Metro Station"],
  },
  {
    "area_name_en": "Zaabeel First",
    "nearest_metro_en": ["Al Jafiliya Metro Station", "Unknown"],
  },
  {
    "area_name_en": "Zaabeel Second",
    "nearest_metro_en": [
      "Financial Centre",
      "Buj Khalifa Dubai Mall Metro Station",
    ],
  },
]

st.markdown(
    """
    <style>
    /* Custom select box focus styling */
    .custom-select:focus {
        border: 2px solid green !important;
        outline: none;
    }
    </style>
    <script>
    // JavaScript to apply 'custom-select' style class to all select elements
    const selectElements = document.querySelectorAll('div[data-testid="stSelectbox"] select');
    selectElements.forEach((el) => {
        el.classList.add('custom-select');
    });
    </script>
    """,
    unsafe_allow_html=True
)

st.title("Property Price Estimator")

# Inject custom CSS and JavaScript for styling select boxes


# Define options and areas data
transaction_types = ['Sales', 'Mortgages']
property_types = ["Unit", "Villa", "Land", "Building"]
property_usages = [
    "Commercial", "Residential", "Hospitality", "Industrial", 
    "Agricultural", "Multi-Use", "Storage", "Residential / Commercial", "Other"
]
areas = [
    {"area_name_en": "Al Barsha First", "nearest_metro_en": ["Mall of the Emirates", "Mashreq"]},
    {"area_name_en": "Downtown Dubai", "nearest_metro_en": ["Burj Khalifa/Dubai Mall", "Business Bay"]}
]
rooms_values = ["Studio", "1 B/R", "2 B/R", "3 B/R", "Office", "Others"]

# Create Streamlit selectboxes
trans_group = st.selectbox('Transaction Type', transaction_types)
property_type_en = st.selectbox('Property Type', property_types)
property_usage_en = st.selectbox('Property Usage', property_usages)
area_name_en = st.selectbox("Location", [area["area_name_en"] for area in areas])

# Handle dependent select box for nearest metro
selected_location = next((area for area in areas if area["area_name_en"] == area_name_en), None)
if selected_location:
    nearest_metro = st.selectbox("Nearest Metro", selected_location["nearest_metro_en"])
else:
    nearest_metro = None

rooms_value = st.selectbox('Rooms', rooms_values)
procedure_area = st.number_input('Property Area', min_value=0.0, step=1.0)

# Select input options
# trans_group = st.selectbox('Transaction Type', ['Sales', 'Mortgages'])
# property_type_en = st.selectbox('Property Type', ["Unit", "Villa", "Land", "Building"])
# property_usage_en = st.selectbox('Property Usage', [
#     "Commercial", "Residential", "Hospitality", "Industrial", 
#     "Agricultural", "Multi-Use", "Storage", "Residential / Commercial", "Other"
# ])
# area_name_en = st.selectbox("Location", [area["area_name_en"] for area in areas])

# selected_location = next((area for area in areas if area["area_name_en"] == area_name_en), None)

# nearest_metro = None
# if selected_location:
#     nearest_metro = st.selectbox("Nearest Metro", selected_location["nearest_metro_en"])


# rooms_value = st.selectbox('Rooms', ["Studio", "1 B/R", "2 B/R", "3 B/R", "Office", "Others"])

# procedure_area = st.number_input('Property Area', min_value=0.0, step=1.0)



# has_parking = st.selectbox('Has Parking?', ['Yes', 'No',])


options = ["Yes", "No"]

# Inject HTML and CSS for a custom-styled select box
st.markdown("""
    <style>
    .custom-select {
        font-size: 16px;
        padding: 8px;
        border-radius: 4px;
        width: 100%;
        border: 1px solid #ccc;
        outline: none;
    }
    .custom-select:focus {
        border: 2px solid green; /* Green border on focus */
    }
    </style>
    <label for="hasParking">Has Parking?</label>
    <select id="hasParking" class="custom-select">
        <option value="Yes">Yes</option>
        <option value="No">No</option>
    </select>
    """, unsafe_allow_html=True)

# Capture the selected option with Streamlit's selectbox to use the value in Streamlit app logic
has_parking = st.selectbox("Select Has Parking", options, key="parking_selectbox")
st.write(f"You selected: {has_parking}")


# Prediction
if st.button('Predict Price'):
    try:
        has_parking = 1 if has_parking == 'Yes' else 0

        # Prepare the input query
        input_data = pd.DataFrame({
             'property_usage_en': [property_usage_en],
              'property_type_en': [property_sub_type_en],
               '"area_name_en"': ["area_name_en"],
                '"nearest_metro_en"': ["nearest_metro_en"],
          
            'rooms_value': [rooms_en],
            'has_parking': [has_parking],
            'procedure_area': [procedure_area],
            'trans_group_en': [trans_group],
        })

        # input_data.reset_index(drop=True, inplace=True)

        # Apply the encoder to the input query
        query_encoded = encoder.transform(input_data)

        # Align features with model's expected input
        query_encoded = query_encoded.reindex(columns=rf.feature_names_in_, fill_value=0)
        
          # Debugging: Print shape of the transformed data

        # Make prediction
        prediction = np.exp(rf.predict(query_encoded))

        st.markdown(f"<h1 style='font-size:20px;'>Predicted price for this property is AED {int(prediction):,} </h1>",
    unsafe_allow_html=True)


    except Exception as e:
        st.error(f"An error occurred: {e}")
       

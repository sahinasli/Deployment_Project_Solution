import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
import pickle

st.markdown("")

# HTML template for the title
html_temp = """
<div style="background-color:tomato;padding:10px">
<h2 style="color:white;text-align:center;">Streamlit ML Cloud App </h2>
</div>
"""
# Display the title
st.markdown(html_temp, unsafe_allow_html=True)
st.markdown("")

# Buttons layout
buttons_row = st.columns([1, 1, 1])  # Adjusting column widths

# Initialize session_state if it's the first run
if 'data_button_clicked' not in st.session_state:
    st.session_state.data_button_clicked = False

# Data button
data_button_key = "data_button"  # Unique key
if buttons_row[0].button("Data", key=data_button_key):
    st.session_state.data_button_clicked = not st.session_state.data_button_clicked

# Initialize 'info_button_clicked' and 'links_button_clicked' if it's the first run
if 'info_button_clicked' not in st.session_state:
    st.session_state.info_button_clicked = False
if 'links_button_clicked' not in st.session_state:
    st.session_state.links_button_clicked = False

# Data display
if st.session_state.data_button_clicked:
    # Data descriptions
    data_descriptions = {
        'make_model': 'Car model',
        'gearbox': 'Gearbox type',
        'drivetrain': 'Drivetrain type',
        'power_kW': 'Engine power (kW)',
        'age': 'Age',
        'empty_weight': 'Empty weight',
        'mileage': 'Mileage',
        'gears': 'Number of gears',
        'cons_avg': 'Average consumption',
        'co_emissions': 'CO emissions',
    }

    # Data as a DataFrame
    data_df = pd.DataFrame(list(data_descriptions.items()), columns=['Variable', 'Description'])

    # Show data descriptions
    st.markdown("---")
    st.dataframe(data_df)
    st.markdown("\n\n")

# Buttons
info_button, links_button = st.columns([1, 0.111])  # Adjusting column widths

# Style for centering buttons
button_style = "text-align: center;"

# Information button
info_button_key = "info_button"  # Unique key
if info_button.button("Information", key=info_button_key):
    st.session_state.info_button_clicked = not st.session_state.info_button_clicked

# Links button
links_button_key = "links_button"  # Unique key
if links_button.button("Links", key=links_button_key):
    st.session_state.links_button_clicked = not st.session_state.links_button_clicked

# Information markdown
info_markdown = """
This project revolves around using machine learning algorithms to estimate car prices. 

The following regression algorithms were implemented:
- Linear Regression
- Lasso Regression
- Ridge Regression
- Decision Tree
- Random Forest
- XGBoost

Model evaluation, grid-search and cross-validation were performed, resulting in the following scores:

| Model             | R2    | MAE      | RMSE     | MAPE  |
| ----------------- | ----- | -------- | -------- | ----- |
| XGBoost           | 0.917 | 1738.874 | 2611.473 | 0.131 |
| Random Forest     | 0.904 | 1919.038 | 2814.804 | 0.153 |
| Lasso             | 0.836 | 2474.291 | 3670.117 | 0.216 |
| Linear Regression | 0.838 | 2479.831 | 3651.842 | 0.219 |
| ElasticNet        | 0.836 | 2460.996 | 3667.654 | 0.212 |
| Decision Tree     | 0.813 | 2720.283 | 3918.682 | 0.221 |
"""

# Links HTML
links_html = """
<div style="margin-bottom: 20px;">
    <h3 style="background-color: #FF6961; color: white; padding: 10px; border-radius: 5px;"> Auto Analytics: Advanced Estimation & Deployment </h3>
    <ul style="list-style-type: none; padding: 0;">
        <li style="margin-bottom: 10px;">
            <a style="color: red;" href="https://github.com/sahinasli?tab=repositories" target="_blank">
                <b>Github Notebook Link</b>
            </a>
        </li>
        <li style="margin-bottom: 10px;">
            <a style="color: yellow;" href="https://nbviewer.org/github/sahinasli/aws_deployment_DA8115/tree/main/" target="_blank">
                <b>Nbviewer Notebook</b>
            </a>
        </li>
        <li style="margin-bottom: 10px;">
            <a style="color: green;" https://www.linkedin.com/in/sahin-asli/" target="_blank">
                <b>My Linkedin Account</b>
            </a>
        </li>
        <li style="margin-bottom: 10px;">
           <a style="color: orange;" href="https://share.streamlit.io/" target="_blank">
                <b>Streamlit Live</b>
            </a>
        </li>
    </ul>
</div>
"""

# Show Information
if st.session_state.info_button_clicked:
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; padding: 10px; border: 1px solid #e0e0e0; border-radius: 10px;'>{info_markdown}</div>", unsafe_allow_html=True)
    st.markdown("\n\n")

# Show Links
if st.session_state.links_button_clicked:
    st.markdown("---")
    st.markdown(f"<div style='text-align: center; padding: 10px; border: 1px solid #e0e0e0; border-radius: 10px;'>{links_html}</div>", unsafe_allow_html=True)
    st.markdown("\n\n")



html_temp_2 = """
<div style ="margin-top:20px"> <img src="https://w7.pngwing.com/pngs/64/222/png-transparent-car-transport-cartoon-car-child-vehicle-cartoon-car-thumbnail.png"> </div>
"""
st.markdown(html_temp_2,unsafe_allow_html=True)



st.markdown("---")

# title of the sidebar
html_temp = """
<div style="background-color:green;padding:10px">
<h2 style="color:white;text-align:center;">Car Price Prediction </h2>
</div>"""

st.sidebar.markdown(html_temp,unsafe_allow_html=True)


selected_algorithm = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "XGBoost"], index = 0)

# Load the appropriate CSV file for the selected algorithm
if selected_algorithm == "Random Forest":
    df = pd.read_csv("rf_data.csv")
    # data_filename = "rf_data.csv"
elif selected_algorithm == "XGBoost":
    df = pd.read_csv("xgb_data.csv")
    
else:
    st.error("Invalid Selection!")


st.header("Training Dataframe is below")
st.markdown("---")
st.write(df.sample(5))


make_model = st.sidebar.selectbox("Select Auto Brand - Model", ['Mercedes-Benz A 160',
 'Mercedes-Benz EQE 350',
 'Mercedes-Benz A 45 AMG',
 'Mercedes-Benz A 35 AMG',
 'Mercedes-Benz C 160',
 'Mercedes-Benz CLA 180',
 'Mercedes-Benz A 220',
 'Mercedes-Benz A 200',
 'Mercedes-Benz A 250',
 'Mercedes-Benz A 180',
 'Mercedes-Benz GLC 220',
 'Mercedes-Benz E 220',
 'Mercedes-Benz B 200',
 'Mercedes-Benz A 150',
 'Mercedes-Benz V 220',
 'Mercedes-Benz AMG GT',
 'Mercedes-Benz EQS',
 'Mercedes-Benz A 140',
 'Mercedes-Benz CL',
 'Mercedes-Benz B 180',
 'Mercedes-Benz GLB 200',
 'Mercedes-Benz E 350',
 'Mercedes-Benz CLA 200',
 'Mercedes-Benz GLA 180',
 'Mercedes-Benz Viano',
 'Mercedes-Benz E 53 AMG',
 'Mercedes-Benz SLK 200',
 'Mercedes-Benz GLE 350',
 'Mercedes-Benz 220',
 'Mercedes-Benz E 400',
 'Mercedes-Benz SLC 43 AMG',
 'Mercedes-Benz SL 350',
 'Mercedes-Benz SLC 250',
 'Mercedes-Benz SLK 350',
 'Mercedes-Benz SLS',
 'Mercedes-Benz CLK 200',
 'Mercedes-Benz C 400',
 'Mercedes-Benz SL 500',
 'Mercedes-Benz SL 65 AMG',
 'Mercedes-Benz C 43 AMG',
 'Mercedes-Benz C 63 AMG',
 'Mercedes-Benz SLK 250',
 'Mercedes-Benz SLK 300',
 'Mercedes-Benz SLC 200',
 'Mercedes-Benz SLK 55 AMG',
 'Mercedes-Benz S 63 AMG',
 'Mercedes-Benz E 500',
 'Mercedes-Benz SLK',
 'Mercedes-Benz SL 55 AMG',
 'Mercedes-Benz CLK 55 AMG',
 'Mercedes-Benz C 300',
 'Mercedes-Benz SLK 280',
 'Mercedes-Benz E 450',
 'Mercedes-Benz SLC 180',
 'Mercedes-Benz E 300',
 'Mercedes-Benz C 200',
 'Mercedes-Benz E 200',
 'Mercedes-Benz SL 63 AMG',
 'Mercedes-Benz S 560',
 'Mercedes-Benz C 180',
 'Mercedes-Benz C 250',
 'Mercedes-Benz C 220',
 'Mercedes-Benz E 250',
 'Mercedes-Benz SL 600',
 'Mercedes-Benz 250',
 'Mercedes-Benz CLK 350',
 'Mercedes-Benz SLC 300',
 'Mercedes-Benz S 500',
 'Mercedes-Benz SLR',
 'Mercedes-Benz SL 400',
 'Mercedes-Benz CLK 280',
 'Mercedes-Benz CLK 240',
 'Mercedes-Benz 200',
 'Mercedes-Benz SLK 230',
 'Mercedes-Benz CL 500',
 'Mercedes-Benz CLA 220',
 'Mercedes-Benz CLA 45 AMG',
 'Mercedes-Benz CLS 500',
 'Mercedes-Benz S 400',
 'Mercedes-Benz CL 63 AMG',
 'Mercedes-Benz CLS 350',
 'Mercedes-Benz GLC 43 AMG',
 'Mercedes-Benz CLS 400',
 'Mercedes-Benz CLK 63 AMG',
 'Mercedes-Benz CLS 250',
 'Mercedes-Benz CLK 500',
 'Mercedes-Benz CLA 35 AMG',
 'Mercedes-Benz GLC 63 AMG',
 'Mercedes-Benz CLS 320',
 'Mercedes-Benz CLS 450',
 'Mercedes-Benz CLS 53 AMG',
 'Mercedes-Benz CL 600',
 'Mercedes-Benz GLE 43 AMG',
 'Mercedes-Benz CLA 250',
 'Mercedes-Benz GLC 200',
 'Mercedes-Benz GLE 63 AMG',
 'Mercedes-Benz GLC 400',
 'Mercedes-Benz CLS',
 'Mercedes-Benz GLE 400',
 'Mercedes-Benz CLK',
 'Mercedes-Benz GLC 250',
 'Mercedes-Benz CLK 320',
 'Mercedes-Benz S 65 AMG',
 'Mercedes-Benz GLC 350',
 'Mercedes-Benz CLS 300',
 'Mercedes-Benz GLE 300',
 'Mercedes-Benz E 50 AMG',
 'Mercedes-Benz S 450',
 'Mercedes-Benz GLC 300',
 'Mercedes-Benz GLE 450',
 'Mercedes-Benz ML 320',
 'Mercedes-Benz ML 63 AMG',
 'Mercedes-Benz ML 500',
 'Mercedes-Benz GL 500',
 'Mercedes-Benz GLE 580',
 'Mercedes-Benz GLE 53 AMG',
 'Mercedes-Benz EQA',
 'Mercedes-Benz G 400',
 'Mercedes-Benz GL 63 AMG',
 'Mercedes-Benz ML 450',
 'Mercedes-Benz ML 300',
 'Mercedes-Benz GLS 400',
 'Mercedes-Benz ML 350',
 'Mercedes-Benz EQC 400',
 'Mercedes-Benz R 300',
 'Mercedes-Benz GLA 45 AMG',
 'Mercedes-Benz GLK 350',
 'Mercedes-Benz GLK 220',
 'Mercedes-Benz GL 350',
 'Mercedes-Benz GLB 220',
 'Mercedes-Benz G 63 AMG',
 'Mercedes-Benz GLA 220',
 'Mercedes-Benz GLK 250',
 'Mercedes-Benz G 350',
 'Mercedes-Benz GLA 250',
 'Mercedes-Benz G 500',
 'Mercedes-Benz GLE 250',
 'Mercedes-Benz ML 400',
 'Mercedes-Benz EQB 350',
 'Mercedes-Benz ML 250',
 'Mercedes-Benz GLA 200',
 'Mercedes-Benz GLA 35 AMG',
 'Mercedes-Benz GLK 200',
 'Mercedes-Benz G',
 'Mercedes-Benz G 55 AMG',
 'Mercedes-Benz GLS 350',
 'Mercedes-Benz GL 420',
 'Mercedes-Benz GLE 500',
 'Mercedes-Benz EQB 300',
 'Mercedes-Benz GLB 250',
 'Mercedes-Benz GLB 180',
 'Mercedes-Benz EQA 250',
 'Mercedes-Benz GLB 35 AMG',
 'Mercedes-Benz X 250',
 'Mercedes-Benz GL 320',
 'Mercedes-Benz GLS 63 AMG',
 'Mercedes-Benz 170',
 'Mercedes-Benz ML 280',
 'Mercedes-Benz G 65 AMG',
 'Mercedes-Benz Sprinter',
 'Mercedes-Benz C 350',
 'Mercedes-Benz E 43 AMG',
 'Mercedes-Benz CLS 63 AMG',
 'Mercedes-Benz C 32 AMG',
 'Mercedes-Benz E 63 AMG',
 'Mercedes-Benz V 300',
 'Mercedes-Benz E 280',
 'Mercedes-Benz EQE 43',
 'Mercedes-Benz V 250',
 'Mercedes-Benz Vito',
 'Mercedes-Benz EQV 300',
 'Mercedes-Benz B 220',
 'Mercedes-Benz B 250',
 'Mercedes-Benz S 300',
 'Mercedes-Benz S 580',
 'Mercedes-Benz S 600',
 'Mercedes-Benz C 280',
 'Mercedes-Benz S 350',
 'Mercedes-Benz CLS 280',
 'Mercedes-Benz S 55 AMG',
 'Mercedes-Benz E 240',
 'Mercedes-Benz C 320',
 'Mercedes-Benz E 320',
 'Mercedes-Benz E 230',
 'Mercedes-Benz EQE 500',
 'Opel Corsa',
 'Opel Astra',
 'Opel Adam',
 'Opel Corsa-e',
 'Opel Meriva',
 'Opel Karl',
 'Opel Agila',
 'Opel Insignia',
 'Opel Vectra',
 'Opel Antara',
 'Opel Ampera',
 'Opel GT',
 'Opel Cascada',
 'Opel Tigra',
 'Opel Speedster',
 'Opel Crossland X',
 'Opel Grandland',
 'Opel Grandland X',
 'Opel Mokka',
 'Opel Mokka X',
 'Opel Crossland',
 'Opel Mokka-E',
 'Opel Vivaro',
 'Opel Zafira Tourer',
 'Opel Zafira Life',
 'Opel Combo Life',
 'Opel Combo',
 'Opel Zafira',
 'Opel Movano',
 'Renault Megane',
 'Renault Clio',
 'Renault Laguna',
 'Renault Twingo',
 'Renault ZOE',
 'Renault Captur',
 'Renault Twizy',
 'Renault Fluence',
 'Renault Grand Scenic',
 'Renault Avantime',
 'Renault Megane E-Tech',
 'Renault Espace',
 'Renault Kangoo Z.E.',
 'Renault Alpine A110',
 'Renault Wind',
 'Renault Coupe',
 'Renault Talisman',
 'Renault Kangoo',
 'Renault Arkana',
 'Renault Kadjar',
 'Renault Alaskan',
 'Renault Koleos',
 'Renault Scenic',
 'Renault Trafic',
 'Renault Master',
 'Renault Express',
 'Renault Grand Espace',
 'Renault P 1400',
 'Renault Latitude',
 'Renault R 9',
 'Renault Grand Modus',
 'Renault R 11',
 'Peugeot 308',
 'Peugeot 206',
 'Peugeot 208',
 'Peugeot 207',
 'Peugeot 1007',
 'Peugeot 307',
 'Peugeot 108',
 'Peugeot Rifter',
 'Peugeot e-208',
 'Peugeot 3008',
 'Peugeot 508',
 'Peugeot Partner',
 'Peugeot 107',
 'Peugeot 106',
 'Peugeot iOn',
 'Peugeot Expert',
 'Peugeot 407',
 'Peugeot RCZ',
 'Peugeot 406',
 'Peugeot 2008',
 'Peugeot 4007',
 'Peugeot 4008',
 'Peugeot 5008',
 'Peugeot e-2008',
 'Peugeot Traveller',
 'Peugeot Bipper',
 'Peugeot Ranch',
 'Peugeot 607',
 'Peugeot 301',
 'Peugeot Boxer',
 'Fiat 500 Abarth',
 'Fiat 595 Abarth',
 'Fiat 500',
 'Fiat Tipo',
 'Fiat Stilo',
 'Fiat 500X',
 'Fiat 500e',
 'Fiat Punto',
 'Fiat Panda',
 'Fiat Fiorino',
 'Fiat 500C',
 'Fiat New Panda',
 'Fiat Grande Punto',
 'Fiat Punto Evo',
 'Fiat Seicento',
 'Fiat 124 Spider',
 'Fiat Barchetta',
 'Fiat Spider Europa',
 'Fiat Fullback',
 'Fiat Freemont',
 'Fiat 500L',
 'Fiat Strada',
 'Fiat Sedici',
 'Fiat Talento',
 'Fiat Croma',
 'Fiat Qubo',
 'Fiat Doblo',
 'Fiat Bravo',
 'Fiat Multipla',
 'SEAT Leon',
 'SEAT Ibiza',
 'SEAT Toledo',
 'SEAT Cordoba',
 'SEAT Arona',
 'SEAT Mii',
 'SEAT Altea XL',
 'SEAT Tarraco',
 'SEAT Arosa',
 'SEAT Ateca'])
gearbox = st.sidebar.selectbox("Select Gearbox", ['Manual', 'Automatic', 'Semi-automatic'])# array i listeye donustur unique degerleri liste olarak yazin
drivetrain = st.sidebar.selectbox("Select Drivetrain", ['Front', '4WD', 'Rear'])
gears = st.sidebar.selectbox("Enter Gears", [1.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
# List of consumption values
consumption_values = [
    3.8, 2.359, 8.4, 7.3, 4.9, 8.2, 6.1, 0.0, 5.4, 6.9, 6.0, 4.5, 5.2, 3.9, 
    3.7, 5.0, 7.1, 7.6, 5.3, 5.6, 1.4, 4.7, 4.0, 4.2, 4.3, 4.1, 4.6, 7.2, 4.4, 
    6.2, 5.5, 6.6, 6.7, 12.4, 9.1, 11.3, 8.9, 9.3, 8.3, 9.8, 3.6, 3.5, 9.6, 
    6.4, 12.9, 1.0, 4.8, 4.35, 5.8, 1.3, 5.7, 8.0, 7.9, 5.1, 5.9, 6.3, 
    7.8, 11.4, 7.8, 6.8, 13.0, 10.1, 13.2, 9.0, 8.6, 12.3, 12.5, 15.1, 
    8.8, 11.5, 9.5, 9.2, 14.0, 11.7, 11.6, 11.9, 13.5, 9.4, 8.7, 11.1, 
    10.4, 7.5, 13.9, 10.2, 7.05, 10.6, 9.7, 4.75, 7.7, 6.5, 7.0, 10.0, 
    8.1, 12.7, 14.1, 9.9, 8.65, 12.1, 10.7, 10.3, 10.9, 12.0, 4.65, 
    14.9, 13.1, 13.4, 6.25, 14.2, 16.4, 9.15, 10.8, 14.3, 10.75, 10.5, 
    11.2, 0.65, 6.85, 8.5, 5.35, 7.4, 2.5, 6.05, 1.9, 0.9, 11.8, 16.5, 
    13.8, 15.9, 3.3, 3.15, 6.7, 12.2, 14.4, 2.0, 12.6, 11.0, 1.8, 2.1, 
    6.35, 0.6, 1.6, 8.25, 1.5, 1.7, 14.5, 2.8, 0.7, 4.15, 2.2, 1.2, 0.8, 
    9.25, 1.1, 3.2, 3.4, 7.25, 5.8, 4.95, 3.95, 5.25, 3.1, 5.05, 3.0, 
    7.3, 4.05, 5.65, 4.85, 4.55, 16.0, 5.75, 5.45, 3.75, 8.55, 5.15, 0.5, 
    6.15
]

# Sort the list in ascending order
sorted_consumption_values = sorted(consumption_values)

# Use sorted list in Streamlit selectbox
cons_avg = st.sidebar.selectbox("Enter Consumption", sorted_consumption_values)

# Display selected consumption value
st.write(f"Selected Consumption: {cons_avg}")
# List of CO emissions
co_emissions_list = [
    98.0, 0.0, 162.0, 169.0, 196.0, 134.0, 141.0, 168.0, 101.5, 22.0, 
    120.0, 138.0, 99.5, 126.0, 140.0, 132.0, 179.0, 121.0, 128.0, 136.0, 
    127.0, 129.0, 173.0, 124.0, 171.0, 133.0, 183.0, 210.0, 130.0, 156.0, 
    123.0, 92.0, 284.0, 209.0, 257.0, 215.0, 192.0, 228.0, 165.0, 145.0, 
    89.0, 161.0, 131.0, 105.0, 188.0, 258.0, 102.0, 175.0, 114.0, 143.0, 
    139.0, 118.0, 164.0, 122.0, 108.0, 103.0, 99.0, 107.0, 1.0, 153.0, 
    104.0, 111.0, 119.0, 101.0, 157.0, 146.0, 242.0, 203.0, 170.0, 259.0, 
    260.0, 178.0, 298.0, 167.0, 308.0, 214.0, 151.0, 292.0, 296.0, 332.0, 
    262.0, 142.0, 218.0, 230.0, 261.0, 281.0, 125.0, 272.0, 195.0, 190.0, 
    263.0, 194.0, 223.0, 324.0, 219.0, 155.0, 227.0, 199.0, 247.0, 255.0, 
    154.0, 244.0, 144.0, 176.0, 236.0, 293.0, 160.0, 233.0, 172.0, 135.0, 
    221.0, 208.0, 166.0, 152.0, 226.0, 149.0, 189.0, 181.0, 148.0, 150.0, 
    116.0, 201.0, 235.0, 246.0, 177.0, 330.0, 185.0, 137.0, 163.0, 158.0, 
    216.0, 211.0, 328.0, 253.0, 275.0, 237.0, 222.0, 304.0, 204.0, 326.0, 
    197.0, 187.0, 290.0, 288.0, 300.0, 205.0, 277.0, 289.0, 280.0, 152.5, 
    224.0, 299.0, 200.0, 217.0, 274.0, 178.5, 125.5, 338.0, 202.0, 248.0, 
    147.0, 340.0, 249.0, 269.0, 184.0, 254.0, 159.0, 24.0, 206.0, 180.0, 
    157.5, 198.0, 240.0, 31.0, 186.0, 191.0, 232.0, 252.0, 282.0, 267.0, 
    225.0, 265.0, 279.0, 212.0, 138.5, 106.0, 250.0, 23.0, 115.0, 52.0, 
    59.0, 276.0, 319.0, 273.0, 238.0, 310.0, 220.0, 193.0, 264.0, 327.0, 
    283.0, 213.0, 48.0, 377.0, 241.0, 28.0, 295.0, 117.0, 239.0, 322.0, 
    169.5, 278.0, 234.0, 207.0, 341.0, 270.0, 174.0, 190.5, 344.0, 373.0, 
    182.0, 243.0, 110.0, 348.0, 251.0, 54.0, 271.0, 229.0, 113.0, 49.0, 
    109.0, 112.0, 96.0, 94.0, 15.0, 13.0, 14.0, 100.0, 42.0, 34.0, 44.0, 
    345.0, 36.0, 33.0, 39.0, 65.0, 261.5, 245.0, 30.0, 268.0, 20.0, 17.0, 
    256.0, 38.0, 317.0, 37.0, 16.0, 216.5, 37.5, 18.0, 106.5, 88.0, 87.0, 
    93.0, 85.0, 91.0, 95.0, 97.0, 82.0, 203.5, 32.0, 35.0, 90.0, 25.0, 
    26.0, 87.5, 133.5, 116.5, 83.0, 118.5, 266.0, 197.5, 131.5, 222.5, 
    29.0, 121.5, 63.5, 27.0, 56.5, 285.0, 94.5, 86.0, 149.5, 79.0, 80.0, 
    146.5, 41.0, 141.5, 7.0, 112.5, 46.0, 104.5, 110.5, 3.0, 123.5, 103.5, 
    107.5, 146.75, 95.5, 126.5, 8.0
]

# Sort the list in ascending order
sorted_co_emissions = sorted(co_emissions_list)

# Use sorted list in Streamlit selectbox
co_emissions = st.sidebar.selectbox("Enter Average CO Emissions", sorted_co_emissions)

# Display selected emission value
st.write(f"Selected CO Emissions: {co_emissions}")





unique_values_list = [
    75.0, 215.0, 310.0, 225.0, 100.0, 90.0, 140.0, 102.0, 66.0, 280.0, 155.0, 85.0, 80.0, 118.0,
    120.0, 143.0, 265.0, 160.0, 125.0, 110.0, 173.0, 70.0, 410.0, 270.0, 320.0, 202.0, 375.0,
    165.0, 132.0, 130.0, 121.0, 115.0, 87.0, 228.0, 67.0, 79.0, 101.0, 142.0, 112.0, 122.0, 81.0,
    195.0, 103.0, 135.0, 235.0, 350.0, 250.0, 245.0, 150.0, 200.0, 285.0, 390.0, 450.0, 287.0,
    232.0, 180.0, 170.0, 185.0, 430.0, 243.0, 335.0, 190.0, 145.0, 210.0, 386.0, 336.0, 212.5,
    220.0, 300.0, 136.0, 105.0, 184.0, 330.0, 99.0, 168.0, 440.0, 139.0, 230.0, 176.0, 154.0,
    95.0, 147.0, 205.0, 55.0, 177.0, 51.0, 152.0, 74.0, 64.0, 63.0, 73.0, 92.0, 96.0, 54.0, 77.0,
    59.0, 43.0, 44.0, 206.0, 104.0, 162.0, 88.0, 65.0, 114.0, 82.0, 194.0, 224.0, 108.0, 231.0,
    76.0, 294.0, 141.0, 191.0, 62.0, 221.0, 134.0, 133.0, 60.0, 61.0, 107.0, 169.0, 128.0, 97.0,
    91.0, 126.5, 129.0, 83.0, 151.0, 56.0, 68.0, 117.0, 52.0, 53.0, 48.0, 50.0, 187.0, 72.0, 57.0,
    86.0, 119.0, 71.0, 78.0, 98.0, 201.0, 192.0, 223.0, 175.0, 213.0, 116.0, 84.0, 163.0, 93.0,
    49.0, 199.0, 182.0, 98.5, 81.5, 113.0
]

# Setting up the number input with the proper min, max, and default values.
power_kw = st.sidebar.number_input(
    "Enter Power (in kW)",
    min_value=min(unique_values_list),  # Minimum value in the list
    max_value=max(unique_values_list),  # Maximum value in the list
    value=unique_values_list[0],        # Default value (first element of the list)
    step=1.0                            # Step size
)
# Define the list of age values
age_values_list = [
    6.0, 0.0, 2.0, 7.0, 17.0, 4.0, 3.0, 9.0, 12.0, 1.0, 8.0, 5.0, 11.0, 16.0, 10.0,
    14.0, 15.0, 18.0, 13.0, 20.0, 19.0
]

# Set up the number input widget in the Streamlit sidebar
age = st.sidebar.number_input(
    "Enter Age",
    min_value=min(age_values_list),      # Minimum value from the list
    max_value=max(age_values_list),      # Maximum value from the list
    value=min(age_values_list),          # Default value (first element of the list)
    step=1.0                             # Step size
)

st.write(f"Selected Age: {age}")
# Assuming the list of empty weights is stored in a DataFrame
empty_weight_values = [1270.0, 2355.0, 1555.0, 1455.0, 1545.0, 1465.0, 1410.0, 1450.0, 1365.0, 1295.0, 1445.0, 1330.0, 
                       1680.0, 1425.0, 1350.0, 1275.0, 1845.0, 1395.0, 1485.0, 1475.0, 1225.0, 1510.0, 2447.0, 1370.0, 
                       1260.0, 1340.0, 1505.0, 1700.0, 1940.0, 1920.0, 1345.0, 1945.0, 2485.0, 2380.0, 1535.0, 1355.0, 
                       1285.0, 1325.0, 2290.0, 1645.0, 2020.0, 1580.0, 1290.0, 1480.0, 1245.0, 1705.0, 1265.0, 1375.0, 
                       1385.0, 1890.0, 1525.0, 1520.0, 1490.0, 1695.0, 1970.0, 1435.0, 2175.0, 1390.0, 1635.0, 1690.0, 
                       2010.0, 1595.0, 1685.0, 1605.0, 1735.0, 1540.0, 1665.0, 1825.0, 1935.0, 1770.0, 1790.0, 1795.0, 
                       1870.0, 1755.0, 1910.0, 1610.0, 1470.0, 2085.0, 1805.0, 1955.0, 1440.0, 1950.0, 1670.0, 1780.0, 
                       1500.0, 2155.0, 1600.0, 1785.0, 1730.0, 1865.0, 1360.0, 1655.0, 1960.0, 1710.0, 2185.0, 1725.0, 
                       1925.0, 2040.0, 1855.0, 1590.0, 1415.0, 1885.0, 1615.0, 1715.0, 1980.0, 2045.0, 1650.0, 2200.0, 
                       1930.0, 1875.0, 1900.0, 1835.0, 1830.0, 1765.0, 1561.0, 1877.0, 1495.0, 2015.0, 1672.0, 2000.0, 
                       2055.0, 1860.0, 2050.0, 1815.0, 1400.0, 1570.0, 1995.0, 2090.0, 1800.0, 1660.0, 2035.0, 1560.0, 
                       1575.0, 2070.0, 1750.0, 2025.0, 1550.0, 1720.0, 1820.0, 2120.0, 1625.0, 2060.0, 2115.0, 2220.0, 
                       2080.0, 1565.0, 1915.0, 2365.0, 1810.0, 1965.0, 1405.0, 2030.0, 2350.0, 1675.0, 1430.0, 1420.0, 
                       1985.0, 1585.0, 2295.0, 2310.0, 1760.0, 2215.0, 2100.0, 1905.0, 2320.0, 2250.0, 2145.0, 2240.0, 
                       2275.0, 2135.0, 2690.0, 2150.0, 2180.0, 2538.0, 2445.0, 2345.0, 2305.0, 2451.0, 2315.0, 2495.0, 
                       2165.0, 2505.0, 2545.0, 2550.0, 2580.0, 2410.0, 2610.0, 2280.0, 2435.0, 2595.0, 2335.0, 2455.0, 
                       3150.0, 2560.0, 2133.0, 2130.0, 2125.0, 2429.0, 2489.0, 2265.0, 1880.0, 2394.0, 2472.0, 2612.0, 
                       1895.0, 2450.0, 2235.0, 2530.0, 1745.0, 2680.0, 1840.0, 2390.0, 1460.0, 2005.0, 2555.0, 1530.0, 
                       1620.0, 2477.0, 2230.0, 2525.0, 2565.0, 2480.0, 2497.0, 2065.0, 2475.0, 2140.0, 1775.0, 2075.0, 
                       2585.0, 1380.0, 1990.0, 2095.0, 2210.0, 2105.0, 1554.0, 2655.0, 2400.0, 1566.0, 2190.0, 1630.0, 
                       2066.0, 2252.0, 2114.0, 1237.0, 1055.0, 1293.0, 1165.0, 1373.0, 1100.0, 1163.0, 1063.0, 1235.0, 
                       1234.0, 1178.0, 1120.0, 1135.0, 1156.0, 1199.0, 1214.0, 1020.0, 1240.0, 1273.0, 1408.0, 1278.0, 
                       1205.0, 1141.0, 1317.0, 1099.0, 1086.0, 939.0, 1033.0, 1437.0, 1233.0, 995.0, 1248.0, 1130.0, 
                       1210.0, 986.0, 1025.0, 1274.0, 1015.0, 1263.0, 1160.0, 1101.0, 1403.0, 1145.0, 1503.0, 1203.0, 
                       1353.0, 1371.0, 1041.0, 910.0, 1010.0, 1341.0, 1513.0, 1736.0, 1337.0, 1168.0, 1591.0, 1308.0, 
                       1198.0, 1387.0, 1406.0, 1733.0, 1716.0, 1701.0, 1816.0, 1843.0, 1306.0, 1731.0, 1601.0, 1714.0, 
                       1393.0, 1438.0, 1331.0, 1426.0, 1397.0, 1533.0, 1567.0, 1487.0, 1572.0, 1547.0, 1258.0, 1251.0, 
                       1711.0, 1488.0, 1220.0, 1180.0, 1401.0, 1447.0, 1297.0, 1479.0, 1497.0, 1369.0, 1300.0, 1207.0, 
                       1287.0, 1328.0, 1200.0, 1169.0, 1126.0, 1272.0, 1023.0, 1243.0, 1040.0, 1257.0, 1194.0, 1208.0, 
                       1164.0, 1124.0, 1036.0, 1227.0, 1244.0, 1277.0, 1312.0, 1381.0, 1231.0, 1396.0, 1423.0, 1159.0, 
                       1482.0, 1196.0, 1123.0, 1356.0, 1304.0, 1283.0, 1334.0, 1378.0, 1324.0, 1338.0, 1367.0, 1195.0, 
                       1185.0, 1462.0, 1202.0, 1478.0, 1372.0, 1384.0, 1457.0, 1288.0, 1090.0, 1323.0, 1402.0, 1452.0, 
                       1363.0, 1388.0, 1239.0, 1299.0, 1302.0, 1386.0, 1394.0, 1422.0, 1316.0, 1242.0, 1424.0, 1222.0, 
                       1221.0, 1416.0, 1315.0, 1269.0, 1241.0, 1267.0, 1289.0, 1292.0, 1436.0, 1391.0, 1320.0, 1339.0, 
                       1476.0, 1303.0, 1412.0, 1493.0, 1279.0, 1504.0, 1451.0, 1439.0, 1443.0, 1327.0, 1322.0, 1421.0, 
                       1351.0, 1417.0, 1419.0, 1377.0, 1442.0, 1474.0, 1499.0, 1491.0, 1362.0, 1411.0, 1494.0, 1374.0, 
                       1511.0, 1431.0, 1333.0, 1506.0, 1508.0, 1498.0, 1527.0, 1379.0, 1509.0, 1383.0, 1432.0, 1528.0, 
                       1366.0, 1529.0, 1502.0, 1538.0, 1336.0, 1473.0, 1516.0, 1534.0, 1382.0, 1352.0, 1364.0, 1433.0, 
                       1407.0, 1532.0, 1409.0, 1398.0, 1389.0, 1413.0, 1380.0, 1496.0, 1404.0, 1481.0, 1518.0, 1483.0, 
                       1531.0, 1383.0, 1444.0, 1471.0, 1468.0, 1392.0, 1467.0, 1456.0, 1466.0, 1461.0, 1427.0, 1429.0, 
                       1418.0, 1515.0, 1507.0, 1519.0, 1521.0, 1514.0, 1484.0, 1536.0, 1501.0, 1523.0, 1522.0, 1526.0, 
                       1500.0, 1449.0, 1448.0, 1524.0, 1453.0, 1492.0, 1512.0, 1490.0, 1517.0, 1486.0, 1463.0, 1526.0]

# Calculate min, max, and mean of the list
min_value = min(empty_weight_values)
max_value = max(empty_weight_values)
default_value = sum(empty_weight_values) / len(empty_weight_values)

# Create the input widget in Streamlit
user_input = st.number_input(
    label="Enter Empty Weight (kg):",
    min_value=min_value,
    max_value=max_value,
    value=default_value,
    step=0.1  # Precision to one decimal place
)

# Display the user input value
st.write(f"Selected Empty Weight: {user_input} kg")

mileage = st.sidebar.selectbox("Enter the Mileage", [1.0,
10.0,
20.0,
25.0,
30.0,
33.0,
50.0,
100.0,
500.0,
533.0,
1000.0,
2000.0,
3000.0,
3001.0,
3021.0,
3033.0,
30481.0,
30535.0,
30600.0,
30708.0,
30803.0,
30890.0,
3094.0,
31000.0,
3115.0,
3127.0,
31500.0,
31543.0,
31615.0,
31915.0,
32000.0,
32252.0,
32256.0,
32300.0,
32334.0,
32534.0,
32613.0,
32668.0,
32800.0,
32838.0,
32946.0,
32980.0,
33080.0,
33141.0,
33297.0,
33349.0,
33400.0,
33576.0,
33648.0,
33750.0,
34000.0,
34100.0,
34408.0,
34438.0,
34483.0,
34626.0,
34766.0,
34942.0,
35000.0,
35100.0,
35342.0,
35421.0,
35552.0,
3561.0,
3569.0,
35758.0,
35800.0,
35850.0,
35900.0,
35900.0,
36000.0,
36021.0,
36063.0,
361241.0,
36200.0,
36250.0,
36430.0,
36450.0,
36500.0,
36600.0,
36605.0,
36700.0,
36800.0,
36838.0,
36930.0,
37000.0,
37021.0,
37063.0,
37100.0,
3722.0,
37300.0,
37450.0,
37800.0,
37870.0,
38000.0,
38000.0,
38082.0,
38123.0,
38216.0,
38236.0,
38408.0,
38500.0,
38547.0,
38556.0,
38600.0,
38800.0,
38827.0,
39000.0,
39082.0,
39100.0,
39267.0,
39313.0,
39600.0,
39680.0,
39757.0,
39900.0,
39945.0,
40000.0,
40000.0,
40100.0,
40200.0,
40335.0,
40400.0,
40505.0,
40698.0,
40814.0,
40835.0,
40900.0,
41000.0,
41004.0,
41096.0,
41266.0,
41272.0,
41276.0,
41300.0,
41500.0,
41591.0,
41600.0,
41700.0,
41731.0,
41800.0,
41806.0,
41876.0,
4195.0,
42000.0,
42157.0,
42200.0,
42300.0,
42356.0,
42400.0,
42452.0,
42500.0,
42650.0,
4278.0,
43000.0,
4303.0,
43150.0,
43200.0,
43290.0,
43300.0,
43355.0,
43400.0,
43447.0,
43500.0,
43600.0,
43700.0,
43800.0,
43819.0,
43900.0,
43987.0,
44000.0,
44200.0,
44266.0,
44300.0,
44319.0,
44400.0,
44500.0,
44770.0,
4500.0,
45000.0,
4537.0,
4539.0,
45400.0,
45701.0,
45927.0,
45999.0,
46000.0,
46041.0,
46200.0,
46300.0,
46403.0,
46437.0,
46499.0,
47000.0,
4700.0,
4715.0,
47200.0,
47400.0,
47500.0,
47514.0,
47700.0,
47800.0,
47890.0,
48000.0,
48040.0,
48100.0,
48243.0,
48400.0,
48500.0,
48600.0,
48700.0,
49000.0,
4900.0,
4921.0,
49222.0,
4939.0,
49400.0,
49444.0,
49500.0,
4955.0,
49600.0,
49627.0,
49658.0,
49700.0,
49800.0,
49900.0,
49927.0,
50000.0,
5000.0,
5000.0,
5000.0,
50340.0,
50400.0,
5058.0,
50600.0,
50700.0,
51000.0,
51078.0,
51197.0,
51200.0,
51450.0,
5147.0,
51551.0,
51800.0,
51813.0,
52000.0,
52019.0,
52100.0,
52200.0,
52400.0,
52497.0,
52565.0,
52694.0,
52745.0,
5266.0,
53000.0,
53200.0,
53258.0,
53300.0,
53400.0,
53600.0,
53650.0,
53700.0,
53800.0,
54118.0,
54400.0,
54468.0,
54500.0,
54510.0,
54600.0,
54750.0,
54800.0,
54900.0,
55000.0,
55100.0,
55118.0,
55300.0,
55363.0,
55500.0,
55538.0,
55600.0,
55700.0,
5576.0,
55800.0,
55900.0,
56000.0,
5609.0,
56100.0,
56232.0,
56300.0,
56387.0,
56400.0,
56600.0,
56666.0,
56700.0,
56800.0,
56876.0,
56896.0,
57000.0,
57050.0,
57100.0,
5712.0,
5726.0,
57300.0,
57400.0,
57500.0,
57700.0,
57740.0,
57750.0,
57795.0,
57800.0,
57900.0,
58000.0,
58100.0,
58200.0,
58300.0,
58500.0,
58517.0,
58592.0,
58600.0,
58700.0,
58800.0,
58802.0,
58805.0,
58900.0,
58981.0,
59000.0,
59012.0,
59100.0,
59191.0,
59300.0,
59381.0,
59500.0,
59579.0,
59600.0,
59618.0,
59800.0,
59828.0,
59900.0,
59950.0,
59964.0,
60000.0,
6000.0,
60100.0,
6010.0,
60211.0,
60248.0,
60300.0,
60400.0,
60500.0,
60600.0,
60700.0,
60800.0,
60900.0,
61000.0,
6100.0,
61100.0,
61170.0,
61200.0,
61300.0,
61400.0,
61474.0,
61500.0,
61564.0,
61600.0,
61618.0,
61700.0,
61800.0,
61900.0,
61926.0,
62000.0,
6209.0,
62100.0,
62200.0,
62207.0,
62400.0,
62400.0,
62500.0,
62600.0,
62700.0,
62796.0,
62800.0,
62866.0,
62900.0,
62948.0,
63000.0,
63019.0,
63036.0,
63096.0,
63130.0,
63200.0,
63267.0,
63300.0,
63400.0,
63500.0,
63522.0,
63600.0,
63675.0,
63700.0,
63800.0,
63871.0,
63900.0,
63990.0,
64000.0,
64100.0,
64200.0,
64400.0,
64450.0,
64500])


# To load machine learning model


model_xgb = pickle.load(open("xgb_pipe_model", "rb"))
model_rf = pickle.load(open("rf_pipe_model", "rb"))


my_dict = {"power_kW":power_kw,
           "age":age,
           "empty_weight": empty_weight,
           "mileage": mileage,
           "gears": gears,
           "cons_avg": cons_avg,
           "co_emissions": co_emissions,
           "make_model": make_model,
           "gearbox": gearbox,
           "drivetrain":drivetrain}

st.header("The values you selected is below")
st.markdown("---")
# Dictionary'i DataFrame'e çevirme
df_input = pd.DataFrame.from_dict([my_dict])

# Sıralama
df_input = df_input[["make_model", "gearbox", "drivetrain", "power_kW", "age", "empty_weight", "mileage", "gears", "cons_avg", "co_emissions"]]

# Tabloyu görüntüleme
st.table(df_input)



st.title("Car Prediction")

# Seçilen make_model'den ilk kelimeyi al ve küçük harfe çevir
make_model_lower = make_model.split()[0].lower()

# Resimleri içeren klasör yolu
pictures_folder = "Picture"

# make_model'e ait resmi bul
png_image_path = os.path.join(pictures_folder, f"{make_model_lower}.png")
jpg_image_path = os.path.join(pictures_folder, f"{make_model_lower}.jpg")

# Resmi görüntüle
try:
    # Önce PNG resmi dene
    image = Image.open(png_image_path)
except FileNotFoundError:
    try:
        # PNG bulunamazsa JPG resmi dene
        image = Image.open(jpg_image_path)
    except FileNotFoundError:
        st.warning(f"Resim bulunamadı: {png_image_path} veya {jpg_image_path}")
        st.stop()

# Resmi 60x60 piksel boyutunda göster
image = image.resize((256, 256))

st.image(image, caption=make_model, use_column_width=80)
# defining the function which will make the prediction using the data
def prediction(model, input_data):
	prediction = model.predict(input_data)
	return prediction

# Making prediction and displaying results
if st.button("Predict"):
    if selected_algorithm == "Random Forest":
        result = prediction(model_rf, df_input)[0]
    else :
        result = prediction(model_xgb, df_input)[0]

try:
    st.success(f"With {selected_algorithm}, Car Price is **{round(result,0)}**")
except NameError:
    st.write("Please press **Predict** button to display the result!")
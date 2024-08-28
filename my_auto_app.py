# Streamlit Documentation: https://docs.streamlit.io/


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # Matplotlib'in pyplot modülünü plt olarak import eder.
import seaborn as sns            # Seaborn'u sns olarak import eder.
import plotly.express as px      # Plotly'nin express modülünü px olarak import eder.
import plotly.graph_objs as go 
from PIL import Image  # to deal with images (PIL: Python imaging library)

# TEXT ELEMENTS
# Title/Text

# Set page layout to wide
st.set_page_config(layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: #166883;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #166883     ;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Homepage", "Data Statistics", "Data Visualization", "Prediction"])

if page == "Homepage":
    st.markdown(
    """
    <h1 style='text-align: center;'>Welcome to the Car Price Prediction App!</h1>
    """,
    unsafe_allow_html=True
    )
  
    img1 = Image.open("images/two_images.jpg")

# Streamlit ile gösterin
    st.image(img1, width=700)
    st.subheader("This application allows you to:")
    st.markdown("""
    - **View data statistics**: Get an overview of the dataset with basic statistics.
    - **Visualize the data**: Explore the data with interactive visualizations.
    - **Make predictions**: Input car details to predict the car's price.
    """)

# Page 1: Data Statistics
elif page == "Data Statistics":
    # Header/Subheader
    st.title("Data Statistics")
    st.header("Autoscout24 Dataset")
# Dataframe
    df = pd.read_csv("Ready_to_ML.csv")

    df_head = df.head()

# To display dataframe there are 3 methods
# Method 1
    st.subheader('')
    st.table(df.head().T)

# Method 2
    st.subheader('First Observations with Selected Features')
    selected_columns = ['make_model', 'body_type', 'fuel_type', 'mileage', 'gears', 'age', 'gearbox', 'price']
    st.write(df[selected_columns].head(7))  # dynamic, you can sort

#st.write(df.info())

# Method 3
    st.subheader('Statistics')
    st.dataframe(df.describe().T)  # dynamic, you can sort

    st.subheader('Showing Makes & Models')
    value_counts = df['make_model'].value_counts()

# Sayısı 20 veya daha fazla olanları filtrele
    filtered_counts = value_counts[value_counts >= 20]

# Streamlit'te göster
    st.dataframe(filtered_counts)

    st.subheader('Showing Fuel Types')
    st.dataframe(df.fuel_type.value_counts()) 

# Page 2: Data Visualization
elif page == "Data Visualization":
    st.title("Data Visualization")

    df = pd.read_csv("Ready_to_ML.csv")

    st.markdown(
    """
    <h3 style='text-align: center; font-size: 24px;'>Price Distribution</h3>
    """,
    unsafe_allow_html=True
)
    

    #st.subheader("Price Distribution")
    plt.figure(figsize=(10, 6))
    sns.histplot(df['price'], kde=True)
    st.pyplot(plt)
    
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
    """
    <h3 style='text-align: center; font-size: 24px;'>Price  vs  Mileage</h3>
    """,
    unsafe_allow_html=True
)
   
    #st.subheader("Price vs. Mileage")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='mileage', y='price', data=df)
    st.pyplot(plt)

    st.markdown("<br>", unsafe_allow_html=True)


    st.markdown(
    """
    <h3 style='text-align: center; font-size: 24px;'>Price Distribution Across Different Car Models</h3>
    """,
    unsafe_allow_html=True
)
    #st.subheader("Price Distribution Across Different Car Models")

    value_counts = df['make_model'].value_counts()

# Sayısı 20 veya daha fazla olanları filtreleme
    filtered_counts = value_counts[value_counts >= 20]

# Filtrelenmiş değerleri alfabetik sıraya göre sıralama
    filtered_models = sorted(filtered_counts.index)

# Kullanıcıdan seçim almak için bir selectbox oluşturma
    selected_model = st.selectbox('**Select a Make and Model:**', filtered_models)

# Grafiği çizmek için fonksiyon tanımla
    def box_strip(model):
        plt.figure(figsize=(10, 6))
        palette = sns.color_palette("Blues", 2)
        sns.boxplot(data=df[df['make_model'] == model],
                x="make_model",
                y="price",
                palette=palette)
    
        sns.stripplot(data=df[df['make_model'] == model],
                  x="make_model",
                  y="price",
                  palette='Set1',
                  dodge=True, alpha=0.7)
    
        plt.title(f"Price Distribution for {model}")
        plt.xlabel("")
        plt.ylabel("Price")
        st.pyplot(plt)

# Seçili modele göre grafiği göster
    if selected_model:
        box_strip(selected_model)

    st.markdown("<br>", unsafe_allow_html=True)    

    st.markdown(
    """
    <h3 style='text-align: center; font-size: 24px;'>Correlation Matrix of Features</h3>
    """,
    unsafe_allow_html=True
)
    #st.subheader("Correlation Matrix of Features")
    plt.figure(figsize=(10,6))
    sns.heatmap(df.corr(numeric_only=True), vmin=-1, vmax=1, annot =True, cmap="Blues");
    st.pyplot(plt)

# Page 3: Prediction
elif page == "Prediction":
    st.title("Prediction")
    # HTML styling
    st.markdown(
        '''
        <style>
            .prediction-input {
                padding: 10px;
                border-radius: 5px;
                margin: 5px 0;
                width: 100%;
            }
        </style>
        ''',
        unsafe_allow_html=True
    )
    import pickle
    filename = "xgb_pipe_model"
    model = pickle.load(open(filename, "rb"))

# To take feature inputs
    st.sidebar.title("Prediction Inputs🚗 ")
    st.subheader("Please enter the necessary details for prediction:")

    make_model_options = ['Dacia Dokker', 'Dacia Duster',
 'Dacia Jogger',
 'Dacia Lodgy',
 'Dacia Logan',
 'Dacia Sandero',
 'Dacia Spring',
 'Fiat 124 Spider',
 'Fiat 500',
 'Fiat 500 Abarth',
 'Fiat 500C',
 'Fiat 500L',
 'Fiat 500X',
 'Fiat 500e',
 'Fiat Panda',
 'Fiat Punto',
 'Fiat Punto Evo',
 'Fiat Tipo',
 'Ford EcoSport',
 'Ford Edge',
 'Ford Fiesta',
 'Ford Focus',
 'Ford Ka/Ka+',
 'Ford Kuga',
 'Ford Mondeo',
 'Ford Mustang',
 'Ford Puma',
 'Ford Ranger',
 'Hyundai BAYON',
 'Hyundai Coupe',
 'Hyundai ELANTRA',
 'Hyundai Genesis',
 'Hyundai IONIQ',
 'Hyundai KONA',
 'Hyundai SANTA FE',
 'Hyundai TUCSON',
 'Hyundai VELOSTER',
 'Hyundai i10',
 'Hyundai i20',
 'Hyundai i30',
 'Hyundai i40',
 'Hyundai iX20',
 'Hyundai iX35',
 'Mercedes-Benz A 180',
 'Mercedes-Benz A 200',
 'Mercedes-Benz A 220',
 'Mercedes-Benz A 250',
 'Mercedes-Benz A 35 AMG',
 'Mercedes-Benz A 45 AMG',
 'Mercedes-Benz C 180',
 'Mercedes-Benz C 200',
 'Mercedes-Benz C 220',
 'Mercedes-Benz C 250',
 'Mercedes-Benz C 300',
 'Mercedes-Benz C 400',
 'Mercedes-Benz C 43 AMG',
 'Mercedes-Benz C 63 AMG',
 'Mercedes-Benz CL',
 'Mercedes-Benz CLA 180',
 'Mercedes-Benz CLA 200',
 'Mercedes-Benz CLA 45 AMG',
 'Mercedes-Benz CLS 350',
 'Mercedes-Benz E 200',
 'Mercedes-Benz E 220',
 'Mercedes-Benz E 250',
 'Mercedes-Benz E 300',
 'Mercedes-Benz E 350',
 'Mercedes-Benz E 400',
 'Mercedes-Benz E 53 AMG',
 'Mercedes-Benz E 63 AMG',
 'Mercedes-Benz GLC 220',
 'Mercedes-Benz GLC 250',
 'Mercedes-Benz GLC 300',
 'Mercedes-Benz GLC 43 AMG',
 'Mercedes-Benz GLC 63 AMG',
 'Mercedes-Benz S 500',
 'Mercedes-Benz SLK 200',
 'Nissan 350Z',
 'Nissan 370Z',
 'Nissan GT-R',
 'Nissan Juke',
 'Nissan Leaf',
 'Nissan Micra',
 'Nissan Murano',
 'Nissan NV200',
 'Nissan Note',
 'Nissan Pulsar',
 'Nissan Qashqai',
 'Nissan Townstar',
 'Nissan X-Trail',
 'Opel Adam',
 'Opel Astra',
 'Opel Cascada',
 'Opel Corsa',
 'Opel Crossland',
 'Opel Crossland X',
 'Opel GT',
 'Opel Grandland',
 'Opel Grandland X',
 'Opel Insignia',
 'Opel Mokka',
 'Opel Mokka X',
 'Opel Tigra',
 'Peugeot 107',
 'Peugeot 108',
 'Peugeot 2008',
 'Peugeot 206',
 'Peugeot 207',
 'Peugeot 208',
 'Peugeot 3008',
 'Peugeot 307',
 'Peugeot 308',
 'Peugeot 407',
 'Peugeot 5008',
 'Peugeot 508',
 'Peugeot RCZ',
 'Renault Alpine A110',
 'Renault Arkana',
 'Renault Captur',
 'Renault Clio',
 'Renault Kadjar',
 'Renault Koleos',
 'Renault Laguna',
 'Renault Megane',
 'Renault Talisman',
 'Renault Twingo',
 'Renault Wind',
 'Renault ZOE',
 'SEAT Arona',
 'SEAT Ateca',
 'SEAT Ibiza',
 'SEAT Leon',
 'SEAT Mii',
 'SEAT Tarraco',
 'Skoda Citigo',
 'Skoda Fabia',
 'Skoda Kamiq',
 'Skoda Karoq',
 'Skoda Kodiaq',
 'Skoda Octavia',
 'Skoda Rapid/Spaceback',
 'Skoda Scala',
 'Skoda Superb',
 'Toyota Auris',
 'Toyota Avensis',
 'Toyota Aygo',
 'Toyota Aygo X',
 'Toyota C-HR',
 'Toyota Corolla',
 'Toyota GT86',
 'Toyota Hilux',
 'Toyota Land Cruiser',
 'Toyota MR 2',
 'Toyota Prius',
 'Toyota RAV 4',
 'Toyota Supra',
 'Toyota Yaris',
 'Toyota Yaris Cross',
 'Volvo C30',
 'Volvo C70',
 'Volvo S60',
 'Volvo S90',
 'Volvo V40',
 'Volvo V40 Cross Country',
 'Volvo V60',
 'Volvo V60 Cross Country',
 'Volvo V90',
 'Volvo V90 Cross Country',
 'Volvo XC40',
 'Volvo XC60',
 'Volvo XC90']

    fuel_type_options = ['Benzine', 'Diesel', 'Liquid/Natural Gas', 'Electric']
    gearbox_options = ["Manual","Automatic","Semi-automatic"]

    make_model = st.sidebar.selectbox('Select Make and Model', make_model_options)
    fuel_type = st.sidebar.selectbox('Select Fuel Type', fuel_type_options)
    gearbox = st.sidebar.selectbox('Select Gear Type', gearbox_options)
    
    mileage = st.sidebar.number_input("mileage:",min_value=0, max_value=67000)
    age = st.sidebar.number_input("age:",min_value=0, max_value=20)
    power_kW = st.sidebar.number_input("power_kW:",min_value=30, max_value=450)
    gears = st.sidebar.number_input("gears:", min_value=1, max_value=8)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"**Selected Make and Model:** {make_model}")
    st.markdown(f"**Selected Fuel Type:** {fuel_type}")
    st.markdown(f"**Selected Gear Type:** {gearbox}")


# Create a dataframe using feature inputs
    my_dict = {"mileage":mileage,
           "age":age,
           "power_kW":power_kW,
           "gears":gears, 
           "make_model": make_model,
           "fuel_type": fuel_type,
           "gearbox": gearbox
           }

    df = pd.DataFrame.from_dict([my_dict])
    html = df.to_html(classes='table table-bordered', index=False)
    html = """
<style>
.table-bordered {
    border: 2px solid black;
    border-collapse: collapse;
}
.table-bordered th, .table-bordered td {
    border: 1px solid black;
    padding: 14px;
}
</style>

<br><br>
""" + df.to_html(classes='table-bordered', index=False)


# HTML'i Streamlit'te göster
    st.markdown(html, unsafe_allow_html=True)


    # HTML styling for button
    st.markdown(
    """
    <style>
    div.stButton > button:first-child {
        #background-color: #317dfa ; /* Green */                          
        background-color: #FF0000; /* Kırmızı */
        color: white;
        padding: 10px 24px;
        font-size: 16px;
        border-radius: 10px;
        border: none;
        cursor: pointer;
    }
    div.stButton > button:first-child:hover {
        background-color: #45a049; /* Darker Green on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Prediction with user inputs
    predict = st.button("Predict")
    if predict:
        result = model.predict(df)
        st.success(f"Predicted Price: €{result[0]:,.2f}")


import sys
import os

# Add the directory containing ObesityType_pipeline to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__name__),
                                             'C:/Users/Tuba/PycharmProjects/pythonProject_Miuul/my_streamlit_frontend')))


import ObesityType_pipeline
import streamlit as st
from datetime import datetime
from streamlit_option_menu import option_menu
import base64
import random
from googleapiclient.discovery import build
from streamlit_extras.stylable_container import stylable_container
import joblib
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.preprocessing import LabelEncoder
# Import the function from your pipeline
from ObesityType_pipeline import encode_column
from sklearn.metrics.pairwise import cosine_similarity

from ObesityType_pipeline import encode_column

from ObesityType_pipeline import (
    obesity_data_prep,
    train_and_evaluate_model,
    plot_confusion_matrix,
    fit_and_evaluate_models,
    predict_and_plot_confusion_matrix,
    plot_feature_importances,
    plot_permutation_importances,
    plot_decision_path,
    plot_partial_dependence,
    encode_column
)


def main():
    # from ObesityType_pipeline import main as pipeline_main

    from dotenv import load_dotenv
    import os

    # .env dosyasını yükleyin
    load_dotenv()

    api_key = os.getenv('api_key')

    # Set page configuration
    st.set_page_config(page_title="Weight Wise App", layout="wide")

    # Mevcut çalışma dizinini alın
    current_dir = os.path.dirname(os.path.abspath(__name__))

    image_paths1 = [
        "obesity.png",
        "scale.png",
        "woman.png",
        "obese_woman_man.png"
    ]
    image_paths2 = [
        "weight_loss_inspiration_1.png",
        "woman_runner.png",
        "weight_loss_inspiration_2.png",
        "weight_loss_journey_3.png",
        "weight_loss.png"
    ]

    image_paths3 = [
        "weight_loss_journey_4.png",
        "woman_runner.png",
        "weight_loss_journey_5.png",
        "weight_loss_inspiration_2.png",
        "gym_members.png"
    ]

    for image_file in image_paths1:
        image_paths1 = [os.path.join(current_dir, 'data', image_file) for image_file in image_paths1]
        #st.image(image_paths1)

    for image_file in image_paths2:
        image_paths2 = [os.path.join(current_dir, 'data', image_file) for image_file in image_paths2]
        #st.image(image_paths1)

    for image_file in image_paths3:
        image_paths3 = [os.path.join(current_dir, 'data', image_file) for image_file in image_paths3]
        #st.image(image_paths1)

    # Function to convert image file to base64
    def image_to_base64(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode()

    # Function to display the carousel
    def display_carousel(image_paths):
        # Convert images to base64
        base64_images = [f"data:image/png;base64,{image_to_base64(path)}" for path in image_paths]

        # Carousel HTML içeriğini oluşturma
        carousel_html = f"""
        <div class="carousel-container" style="width: 60%; margin: auto; position: relative;">
            <div class="carousel" style="width: 100%; overflow: hidden; position: relative;">
                {"".join([f'<img src="{img}" alt="Carousel Image" style="width: 100%; display: none;">' for img in base64_images])}
            </div>
            <script>
                let currentIndex = 0;
                const images = document.querySelectorAll('.carousel img');
                const totalImages = images.length;

                function showNextImage() {{
                    images[currentIndex].style.display = 'none';
                    currentIndex = (currentIndex + 1) % totalImages;
                    images[currentIndex].style.display = 'block';
                }}

                setInterval(showNextImage, 3000);
                images.forEach((img, index) => {{
                    img.style.display = index === 0 ? 'block' : 'none';
                }});
            </script>
        </div>
        """

        # Embed the carousel into Streamlit app
        st.components.v1.html(carousel_html, height=500, scrolling=False)

    # Function to display the home page content
    def home_page():

        st.markdown("""
                <style>
                body {
                    background-color: #f5f5f5;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                }
                .main-content {
                    text-align: center;
                }
                h1 {
                    color: #c30452;
                }
                </style>
                """, unsafe_allow_html=True)

        # Ana içeriği kapsayan div
        st.markdown('<div class="main-content">', unsafe_allow_html=True)

        # st.title("HOME")
        st.markdown("""
                    <h1 style="color:#c30452; text-align: center; ">WEIGHT WISE</h1>

                     """, unsafe_allow_html=True)

        # Resim dosyasını yükleme ve gösterme
        image_path_logo = 'Weight_Wise_Logo.png'

        image_path_logo = os.path.join(current_dir, 'data', image_path_logo)

        image_base64_logo = image_to_base64(image_path_logo)

        # Streamlit uygulaması
        st.markdown(f"""
                                          <div style="text-align: center;">
                                              <img src="data:image/png;base64,{image_base64_logo}" alt="Weight Wise Logo" style="width:30%;"/>
                                              <p></p>
                                          </div>
                                      """, unsafe_allow_html=True)


        def main():
            st.markdown("""
                            <style>
                            body {
                                background-color: #f5f5f5;
                                display: flex;
                                justify-content: center;
                                align-items: center;
                                height: 100vh;
                                margin: 0;
                            }
                            .main-content {
                                text-align: center;
                            }
                            h1 {
                                color: #c30452;
                            }
                            </style>
                            """, unsafe_allow_html=True)

            # Ana içeriği kapsayan div
            st.markdown('<div class="main-content">', unsafe_allow_html=True)

            header = st.container()
            dataset = st.container()
            features = st.container()
            discover = st.container()

            with header:
                # st.title("Welcome to Weight Wise, your smart coach for weight tracking !!!")
                st.markdown("""
                                    <h2 style="color:#c30452;">Welcome to Weight Wise, your smart coach for weight tracking !!!</h2>

                                     """, unsafe_allow_html=True)

                st.markdown("""
                **Welcome to Weight Wise**, your smart coach for weight tracking!

                Here you can:
                - Monitor your Weight, Obesity Level, BMI, BMR, Ideal Weight and Daily Calorie Intake👩🏻‍🔬👨🏻‍🔬🧮
                - Get personalized meal recommendations based on your DCI, Diet Type and Cuisine Type 🍲🍏🍳
                - Stay motivated on your journey to a healthier you with Blog writings, YouTube searches and more 🌸🌟😊
                """)
                st.markdown("""
                                According to **WHO (World Health Organization)** overweight is a condition of excessive fat deposits.
                                Obesity is a chronic complex disease defined by excessive fat deposits that can impair health.
                                Obesity can lead to increased risk of type 2 diabetes and heart disease, it can affect bone health and reproduction, it increases the risk of certain cancers.
                                Obesity influences the quality of living, such as sleeping or moving.
                                The diagnosis of overweight and obesity is made by measuring people’s weight and height and by calculating the body mass index (BMI): weight (kg)/height² (m²).
                                The body mass index is a surrogate marker of fatness and additional measurements, such as the waist circumference, can help the diagnosis of obesity.
                                The BMI categories for defining obesity vary by age and gender in infants, children and adolescents.
                                
                                **Adults**
                                - For adults, WHO defines overweight and obesity as follows:
                                - overweight is a BMI greater than or equal to 25; and
                                - obesity is a BMI greater than or equal to 30.
                                """)


            with dataset:
                # st.header("demographics.csv")
                st.markdown("""
                                    <h2 style="color:#c30452;">Project Overview</h2>

                                                     """, unsafe_allow_html=True)
                #st.text("This dataset is NHANES 2017_2018 dataset, I found this dataset in this link"

                st.markdown("""
                                                In this multiclass classification ML project, obesity levels are estimated based on individuals' daily living habits (e.g., eating patterns, physical activity levels, smoking, family history with overweight) and demographic features (e.g., height, weight, age).
                                                Variables such as BMI, BMR, Ideal Weight, and Daily Calorie Intake are calculated to determine the daily calorie limit. 
                                                Additionally, meal and alternative meal recommendations are provided using a Content-Based Filtering with Nutritional Similarity algorithm based on the individual's diet type and cuisine preferences.
                                                
                                                **The datasets used are ObesityDataSet_raw_and_data_sinthetic.csv and All_Diets.csv from Kaggle.**
                                                - https://www.kaggle.com/code/mpwolke/obesity-levels-life-style
                                                - https://www.kaggle.com/datasets/thedevastator/healthy-diet-recipes-a-comprehensive-dataset/data
                                                """)

            with features:
                # st.header("The features I created")
                st.markdown("""
                                    <h2 style="color:#c30452;">The features I created</h2>

                                                     """, unsafe_allow_html=True)
                st.markdown("""
                                               **BMI :** Body Mass Index (BMI) is calculated by dividing body weight (kg) by the square of height (m), resulting in the Body Mass Index (BMI).
                                               
                                               **BMR :** Basal Metabolic Rate (BMR) calculation helps you determine the amount of calories you need to expend daily. One of the most commonly used methods for calculating BMR is the Harris-Benedict formula. According to the Harris-Benedict formula:
                                                     
                                                     - For an adult male, BMR = 88.362 + (13.397 × weight in kg) + (4.799 × height in cm) – (5.677 × age in years)
                                                   - For an adult female, BMR = 447.593 + (9.247 × weight in kg) + (3.098 × height in cm) – (4.330 × age in years) 
                                                     
                                               **DCI :** Daily Calorie Intake is calculated by multiplying your BMR by your daily activity level (ranging from 0 for never to 3 for always engaging in physical activity).
                                               
                                               **Ideal_Weight :** For calculating Ideal Weight, I used a BMI range of 24-29 for women and men separately, and then multiplied this range by the square of height (m).
                                               
                                               """)


            with discover:
                # st.header("Time to train the model!")
                st.markdown("""
                                    <h2 style="color:#c30452;">Time to discover the Weight Wise app, Enjoy!!! 🐝 </h2>

                                                     """, unsafe_allow_html=True)


        if __name__ == "__main__":
            main()


    # Function to display other pages (Page 1 to Page 6) - unchanged
    def Insights():
        # st.title("Page 1")
        # st.write("Content for Page 1")
        # st.markdown("""
        #                     <h1 style="color:#c30452;">Page 1</h1>
        #
        #                      """, unsafe_allow_html=True)

        def main():
            # st.title("Obesity Type Pipeline")
            st.markdown("""
                            <h1 style="color:#c30452;">Weight Wise Pipeline Insights</h1>

                             """, unsafe_allow_html=True)


            # Resim dosyasını yükleme ve gösterme
            image_path1 = 'visual_1.png'

            # Dinamik dosya yolunu oluşturun
            image_path1 = os.path.join(current_dir, 'data', image_path1)

            image_base64_1 = image_to_base64(image_path1)

            # Streamlit uygulaması
            st.markdown(f"""
                                  <div style="text-align: center;">
                                      <img src="data:image/png;base64,{image_base64_1}" alt="Obesity Type Distribution" style="width:90%;"/>
                                      <p></p>
                                  </div>
                              """, unsafe_allow_html=True)

            image_path2 = 'visual_2_histogram.png'

            image_path2 = os.path.join(current_dir, 'data', image_path2)

            image_base64_2 = image_to_base64(image_path2)

            # Streamlit uygulaması
            st.markdown(f"""
                                  <div style="text-align: center;">
                                      <img src="data:image/png;base64,{image_base64_2}" alt="Histogram" style="width:90%;"/>
                                      <p></p>
                                  </div>
                              """, unsafe_allow_html=True)

            image_path3 = 'visual_3_density.png'

            image_path3 = os.path.join(current_dir, 'data', image_path3)

            image_base64_3 = image_to_base64(image_path3)

            # Streamlit uygulaması
            st.markdown(f"""
                                  <div style="text-align: center;">
                                      <img src="data:image/png;base64,{image_base64_3}" alt="Density" style="width:90%;"/>
                                      <p>Density</p>
                                  </div>
                              """, unsafe_allow_html=True)

            image_path4 = 'visual_4_gender_overweight.png'

            image_path4 = os.path.join(current_dir, 'data', image_path4)

            image_base64_4 = image_to_base64(image_path4)

            # Streamlit uygulaması
            st.markdown(f"""
                                  <div style="text-align: center;">
                                      <img src="data:image/png;base64,{image_base64_4}" alt="Gender_Overweight" style="width:70%;"/>
                                      <p></p>
                                  </div>
                              """, unsafe_allow_html=True)

            image_path5 = 'visual_5_gender_obesity.png'

            image_path5 = os.path.join(current_dir, 'data', image_path5)

            image_base64_5 = image_to_base64(image_path5)

            # Streamlit uygulaması
            st.markdown(f"""
                                   <div style="text-align: center;">
                                       <img src="data:image/png;base64,{image_base64_5}" alt="Gender_Obese" style="width:70%;"/>
                                       <p></p>
                                   </div>
                               """, unsafe_allow_html=True)

            st.markdown("""
                <h2 style="color:#c30452;">Model Best Scores:</h2>
                <ul style="font-size:20px; color:#c30452;">
                        <ul style="font-size:20px; color:#c30452;">
                        <li><b>OvR_KNN: 0.8517031948946452</b></li>
                        <li><b>OvO_KNN: 0.8479406299641796</b></li>
                        <li><b>OvR_DT: 0.9726296179863596</b></li>
                        <li><b>OvO_DT: 0.974704841344139</b></li>
                        <li><b>OvR_LR: 0.7698722914616829</b></li>
                        <li><b>OvO_LR: 0.9539901584563802</b></li>
                    </ul>
                        """, unsafe_allow_html=True)



            image_path6 = 'visual_6_feature_importance.png'

            image_path6 = os.path.join(current_dir, 'data', image_path6)

            image_base64_6 = image_to_base64(image_path6)

            # Streamlit uygulaması
            st.markdown(f"""
                                    <div style="text-align: center;">
                                        <img src="data:image/png;base64,{image_base64_6}" alt="Feature Importance" style="width:90%;"/>
                                        <p></p>
                                    </div>
                                """, unsafe_allow_html=True)

            image_path7 = 'visual_7_permutation_importance.png'

            image_path7 = os.path.join(current_dir, 'data', image_path7)

            image_base64_7 = image_to_base64(image_path7)

            # Streamlit uygulaması
            st.markdown(f"""
                                    <div style="text-align: center;">
                                        <img src="data:image/png;base64,{image_base64_7}" alt="Permutation Importance" style="width:90%;"/>
                                        <p></p>
                                    </div>
                                """, unsafe_allow_html=True)

        if __name__ == "__main__":
            main()

        # if __name__ == "__main__":
        #     main()

    def Prediction():
        st.markdown("""
            <h1 style="color:#c30452;">Weight Wise Prediction</h1>



            """, unsafe_allow_html=True)




        # Define two columns with adjustable weights
        col1, col2 = st.columns([1, 1])  # You can change the ratio to adjust the widths

        # Initialize session state if not already initialized
        if 'prediction_results' not in st.session_state:
            st.session_state.prediction_results = {
                'prediction_label': None,
                'bmi': None,
                'bmr': None,
                'dci': None,
                'ideal_weight': None,
                'difference': None
            }
        with col1:

            st.markdown("""
                       <h4 style="color:#c30452;">Please Enter Your Personal Data for Prediction</h4>


                       """, unsafe_allow_html=True)


            # Reserve space for the button
            button_placeholder = st.empty()
            # button_placeholder2 = st.empty()
            # Create a placeholder for prediction results
            results_placeholder = st.empty()

        # Load the model
        model = joblib.load('ovo_decision_tree_model.pkl')

        # Define the feature columns in the correct order
        feature_columns = ['AGE', 'HEIGHT', 'WEIGHT', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI', 'IDEAL_WEIGHT', 'BMR',
                           'DCI',
                           'GENDER_Male', 'FAMILY_HISTORY_WITH_OVERWEIGHT_yes', 'FAVC_yes',
                           'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no', 'SMOKE_yes', 'SCC_yes',
                           'CALC_Frequently', 'CALC_Sometimes', 'CALC_no', 'MTRANS_Bike',
                           'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking']

        # Define the dictionary for label encoding and decoding
        label_dict = {
            0: 'Insufficient_Weight',
            1: 'Normal_Weight',
            2: 'Obesity_Type_I',
            3: 'Obesity_Type_II',
            4: 'Obesity_Type_III',
            5: 'Overweight_Level_I',
            6: 'Overweight_Level_II'
        }

        # Ideal weight data
        ideal_weight_woman = {
            (13, 19): lambda height: 24 * (height ** 2),
            (20, 29): lambda height: 24 * (height ** 2),
            (30, 39): lambda height: 24.9 * (height ** 2),
            (40, 49): lambda height: 27 * (height ** 2),
            (50, 59): lambda height: 26.5 * (height ** 2),
            (60, 69): lambda height: 28 * (height ** 2),
            (70, 79): lambda height: 29 * (height ** 2)
        }

        ideal_weight_man = {
            (13, 19): lambda height: 24 * (height ** 2),
            (20, 29): lambda height: 24 * (height ** 2),
            (30, 39): lambda height: 24.9 * (height ** 2),
            (40, 49): lambda height: 26 * (height ** 2),
            (50, 59): lambda height: 26.5 * (height ** 2),
            (60, 69): lambda height: 28 * (height ** 2),
            (70, 79): lambda height: 29 * (height ** 2)
        }

        def get_ideal_weight(age, height, is_male):
            if not is_male:  # Female
                for interval, weight in ideal_weight_woman.items():
                    if interval[0] <= age <= interval[1]:
                        if callable(weight):
                            return weight(height)
                        return weight
            else:  # Male
                for interval, weight in ideal_weight_man.items():
                    if interval[0] <= age <= interval[1]:
                        if callable(weight):
                            return weight(height)
                        return weight
            return None

            # Function to calculate BMR

        def calculate_bmr(is_male, weight, height, age):
            if is_male:
                return 88.362 + (13.397 * weight) + (4.799 * (height * 100)) - (5.677 * age)
            else:
                return 447.593 + (9.247 * weight) + (3.098 * (height * 100)) - (4.330 * age)

        # Function to calculate DCI based on BMR and FAF
        def calculate_dci(bmr, faf):
            if faf == 0:
                return 1.2 * bmr
            elif faf == 1:
                return 1.375 * bmr
            elif faf == 2:
                return 1.55 * bmr
            elif faf == 3:
                return 1.725 * bmr
            else:
                return bmr  # default case if FAF is not 0, 1, 2, or 3

        def get_user_input():
            st.sidebar.header("User Input Parameters")
            age = st.sidebar.number_input("Enter age:", min_value=13, max_value=100, value=25, step=1)
            height = st.sidebar.number_input("Enter height in meters:", min_value=0.5, max_value=2.5, value=1.75,
                                             step=0.01)
            weight = st.sidebar.number_input("Enter weight in kg:", min_value=20, max_value=200, value=70, step=1)
            gender = st.sidebar.selectbox("Enter gender:", ["Male", "Female"])
            is_male = gender == 'Male'
            bmi = weight / (height ** 2)
            bmr = calculate_bmr(is_male, weight, height, age)

            ideal_weight = get_ideal_weight(age, height, is_male)
            faf = st.sidebar.number_input("How often do you engage in physical activity? 0 (never) to 3 (always):",
                                          min_value=0, max_value=3, value=1, step=1)
            dci = calculate_dci(bmr, faf)
            st.session_state.dci = dci
            user_input = {
                'AGE': age,
                'HEIGHT': height,
                'WEIGHT': weight,
                'FCVC': st.sidebar.number_input("Consume vegetables? 1 (Never) to 3 (Always):", min_value=1,
                                                max_value=3, value=2, step=1),
                'NCP': st.sidebar.number_input("How many meals does the individual have daily? 1 to 3:", min_value=0,
                                               max_value=3, value=2, step=1),
                'CH2O': st.sidebar.number_input(
                    "How much water do you drink daily? 1 (less than a liter), 2 (1 to 2 liters), or 3 (more than 2 liters): ",
                    min_value=1, max_value=3, value=2, step=1),
                'FAF': faf,
                'TUE': st.sidebar.number_input(
                    "How many hours do you spend sitting on a typical day? 0 (less than an hour), 1 (1 to 2 hours), or 2 (more than 2 hours): ",
                    min_value=0, max_value=2, value=1, step=1),
                'BMI': bmi,
                'BMR': bmr,
                'DCI': dci,
                'IDEAL_WEIGHT': ideal_weight,
                'GENDER_Male': is_male,
                'FAMILY_HISTORY_WITH_OVERWEIGHT_yes': st.sidebar.selectbox("Family history with overweight:",
                                                                           ["yes", "no"]) == 'yes',
                'FAVC_yes': st.sidebar.selectbox("Consume high caloric food frequently:", ["yes", "no"]) == 'yes',
                'CAEC_Frequently': st.sidebar.selectbox("Do you monitor the calories you eat (frequently) :",
                                                        ["yes", "no"]) == 'yes',
                'CAEC_Sometimes': st.sidebar.selectbox("Do you monitor the calories you eat (sometimes) :",
                                                       ["yes", "no"]) == 'yes',
                'CAEC_no': st.sidebar.selectbox("Do you monitor the calories you eat (no) :", ["yes", "no"]) == 'yes',
                'SMOKE_yes': st.sidebar.selectbox("Do you smoke:", ["yes", "no"]) == 'yes',
                'SCC_yes': st.sidebar.selectbox("Do you monitor the calories you burn:", ["yes", "no"]) == 'yes',
                'CALC_Frequently': st.sidebar.selectbox("Do you take extra calories (frequently) :",
                                                        ["yes", "no"]) == 'yes',
                'CALC_Sometimes': st.sidebar.selectbox("Do you take extra calories (sometimes) :",
                                                       ["yes", "no"]) == 'yes',
                'CALC_no': st.sidebar.selectbox("Do you take extra calories (no) :", ["yes", "no"]) == 'yes',
                'MTRANS_Bike': st.sidebar.selectbox("Transportation method is bike:", ["yes", "no"]) == 'yes',
                'MTRANS_Motorbike': st.sidebar.selectbox("Transportation method is motorbike:", ["yes", "no"]) == 'yes',
                'MTRANS_Public_Transportation': st.sidebar.selectbox("Transportation method is public transportation:",
                                                                     ["yes", "no"]) == 'yes',
                'MTRANS_Walking': st.sidebar.selectbox("Transportation method is walking:", ["yes", "no"]) == 'yes'
            }
            return user_input, round(bmi, 2), round(bmr, 0), round(dci, 0), round(ideal_weight, 2), weight

        def preprocess_input(user_input):
            df = pd.DataFrame([user_input])
            df['AGE'] = df['AGE'].astype(int)
            df['HEIGHT'] = df['HEIGHT'].astype(float)
            df['WEIGHT'] = df['WEIGHT'].astype(float)
            df['FCVC'] = df['FCVC'].astype(float)
            df['NCP'] = df['NCP'].astype(float)
            df['CH2O'] = df['CH2O'].astype(int)
            df['FAF'] = df['FAF'].astype(int)
            df['TUE'] = df['TUE'].astype(int)
            df['BMI'] = df['BMI'].astype(float)
            df['BMR'] = df['BMR'].astype(float)
            df['DCI'] = df['DCI'].astype(float)
            df['IDEAL_WEIGHT'] = df['IDEAL_WEIGHT'].astype(float)
            df[[
                'GENDER_Male', 'FAMILY_HISTORY_WITH_OVERWEIGHT_yes', 'FAVC_yes',
                'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no', 'SMOKE_yes', 'SCC_yes',
                'CALC_Frequently', 'CALC_Sometimes', 'CALC_no', 'MTRANS_Bike',
                'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking'
            ]] = df[[
                'GENDER_Male', 'FAMILY_HISTORY_WITH_OVERWEIGHT_yes', 'FAVC_yes',
                'CAEC_Frequently', 'CAEC_Sometimes', 'CAEC_no', 'SMOKE_yes', 'SCC_yes',
                'CALC_Frequently', 'CALC_Sometimes', 'CALC_no', 'MTRANS_Bike',
                'MTRANS_Motorbike', 'MTRANS_Public_Transportation', 'MTRANS_Walking'
            ]].astype(bool)
            return df

        def predict(user_input):
            # Preprocess user input
            df = preprocess_input(user_input)
            df_for_model = df.drop(columns=['IDEAL_WEIGHT', 'BMR', 'DCI'])

            # Make a prediction
            prediction = model.predict(df_for_model)[0]

            # Convert the prediction to a readable format using the dictionary
            prediction_label = label_dict[prediction]
            prediction_dict = {v: k for k, v in label_dict.items()}

            return prediction_label, prediction_dict

        def main():

            # if 'dci' not in st.session_state:
            #     st.session_state.dci = 0  # Default value

            user_input, bmi, bmr, dci, ideal_weight, weight = get_user_input()

            difference = round(weight - ideal_weight, 0)
            if button_placeholder.button("Predict", key="predict_button"):
                prediction_label, prediction_dict = predict(user_input)

                # Simulate prediction and data
                st.session_state.prediction_results['prediction_label'] = prediction_label
                st.session_state.prediction_results['bmi'] = bmi
                st.session_state.prediction_results['bmr'] = bmr
                st.session_state.prediction_results['dci'] = dci
                st.session_state.prediction_results['ideal_weight'] = ideal_weight
                st.session_state.prediction_results['difference'] = difference

                with results_placeholder.container():
                    if st.session_state.prediction_results['prediction_label'] is not None:
                        st.write(
                            f"Your Obesity Type Prediction is : {st.session_state.prediction_results['prediction_label']}")
                        st.write(f"Your Body Mass Index (BMI) is : {st.session_state.prediction_results['bmi']}")
                        st.write(f"Your Basal Metabolic Rate (BMR) is : {st.session_state.prediction_results['bmr']}")
                        st.write(f"Your Daily Calorie Intake (DCI) is : {st.session_state.prediction_results['dci']}")
                        st.write(f"Your Ideal Weight is : {st.session_state.prediction_results['ideal_weight']}")

                        if (st.session_state.prediction_results['difference'] >= 0 and
                                st.session_state.prediction_results['prediction_label'] in (
                                "Normal_Weight", "Insufficent_Weight")):
                            st.write(f"You can put on maximum {difference} kg ")
                        elif (st.session_state.prediction_results['difference'] >= 0 and
                              st.session_state.prediction_results['prediction_label'] not in (
                              "Normal_Weight", "Insufficent_Weight")):
                            st.write(f"You need to lose :  {difference} kg")
                        else:
                            st.write(
                                f"You can put on maximum {abs(st.session_state.prediction_results['difference'])} kg ")

                        st.session_state.dci = dci

        if __name__ == "__main__":
            main()

        def main():

            with col2:
                # st.header("Meal Recommendation System")

                st.markdown("""
                                     <h4 style="color:#c30452;">Meal Recommendation System</h4>


                                     """, unsafe_allow_html=True)

                st.write(f"Your Daily Calorie Intake (DCI) is : {st.session_state.prediction_results['dci']}")
                df = pd.read_csv(
                    "C:/Users/Tuba/PycharmProjects/pythonProject_Miuul/my_streamlit_frontend/data/All_Diets.csv")
                df = df.rename(columns=lambda x: x.upper())
                df.drop_duplicates(inplace=True)
                df.drop(columns=['EXTRACTION_DAY', 'EXTRACTION_TIME'], inplace=True, axis=1)
                df = df[~df['RECIPE_NAME'].str.contains('pork', case=False)]
                df = df[~df['RECIPE_NAME'].duplicated(keep='first')]

                # 1 gr karbonhidrat 4 kcal içerir.
                # 1 gram protein 4 kcal içerir.
                # 1 gram yağ 9 kcal içerir.
                df['TOTAL_CALORIE'] = 4 * df['CARBS(G)'] + 4 * df['PROTEIN(G)'] + 9 * df['FAT(G)']

                df['DIET_TYPE'] = df['DIET_TYPE'].str.title()
                df['CUISINE_TYPE'] = df['CUISINE_TYPE'].str.title()

                # Normalize nutritional values for cosine similarity
                df['PROTEIN(G)_NORM'] = df['PROTEIN(G)'] / df['PROTEIN(G)'].max()
                df['CARBS(G)_NORM'] = df['CARBS(G)'] / df['CARBS(G)'].max()
                df['FAT(G)_NORM'] = df['FAT(G)'] / df['FAT(G)'].max()

                # User inputs
                diet_type = st.selectbox('Select Diet Type', df['DIET_TYPE'].unique())
                cuisine_type = st.selectbox('Select Cuisine Type', df['CUISINE_TYPE'].unique())
                calorie_limit = st.session_state.dci

                # Get recommendations
                if st.button('Get Recommendations', key="recommendation_button"):

                    filtered_df = df[(df['DIET_TYPE'] == diet_type) & (df['CUISINE_TYPE'] == cuisine_type) & (
                            df['TOTAL_CALORIE'] <= calorie_limit)]

                    if not filtered_df.empty:
                        st.write('Recommended Meals:')
                        st.dataframe(filtered_df[['RECIPE_NAME', 'TOTAL_CALORIE', 'PROTEIN(G)', 'CARBS(G)', 'FAT(G)']])

                        # Find similar meals based on nutritional content
                        nutritional_cols = ['PROTEIN(G)_NORM', 'CARBS(G)_NORM', 'FAT(G)_NORM']
                        nutritional_matrix = filtered_df[nutritional_cols].values
                        similarity_matrix = cosine_similarity(nutritional_matrix)

                        # Recommend meals based on similarity
                        top_similar_indices = np.argsort(-similarity_matrix, axis=1)

                        # Display similar meals for the first 5 meals in filtered_df
                        for i, (index, meal) in enumerate(filtered_df.iterrows()):
                            if i >= 5:  # Only process the first 5 meals
                                break

                            st.write(f"Alternatives for {meal['RECIPE_NAME']}:")

                            # Ensure the index is within bounds of the similarity matrix
                            if index >= len(top_similar_indices):
                                st.write("No similar meals found within the given constraints.")
                                continue

                            # Ensure indices are within bounds
                            valid_indices = [idx for idx in top_similar_indices[index][1:4] if idx < len(filtered_df)]

                            if valid_indices:
                                similar_meals = filtered_df.iloc[valid_indices]
                                st.dataframe(
                                    similar_meals[['RECIPE_NAME', 'TOTAL_CALORIE', 'PROTEIN(G)', 'CARBS(G)', 'FAT(G)']]
                                )
                            else:
                                st.write("No similar meals found within the given constraints.")
                        else:
                            st.write('No meals found within the given constraints.')

                        # Show the prediction results if they exist
                    with col1:
                        if st.session_state.prediction_results['prediction_label'] is not None:
                            st.write(
                                f"Your Obesity Type Prediction is : {st.session_state.prediction_results['prediction_label']}")
                            st.write(f"Your Body Mass Index (BMI) is : {st.session_state.prediction_results['bmi']}")
                            st.write(
                                f"Your Basal Metabolic Rate (BMR) is : {st.session_state.prediction_results['bmr']}")
                            st.write(
                                f"Your Daily Calorie Intake (DCI) is : {st.session_state.prediction_results['dci']}")
                            st.write(f"Your Ideal Weight is : {st.session_state.prediction_results['ideal_weight']}")

                            if (st.session_state.prediction_results['difference'] >= 0 and
                                    st.session_state.prediction_results['prediction_label'] in (
                                            "Normal_Weight", "Insufficent_Weight")):
                                st.write(f"You can put on maximum {st.session_state.prediction_results['difference']} kg ")
                            elif (st.session_state.prediction_results['difference'] >= 0 and
                                  st.session_state.prediction_results['prediction_label'] not in (
                                          "Normal_Weight", "Insufficent_Weight")):
                                st.write(f"You need to lose :  {st.session_state.prediction_results['difference']} kg")
                            else:
                                st.write(
                                    f"You can put on maximum {abs(st.session_state.prediction_results['difference'])} kg ")



        if __name__ == "__main__":
            main()

    def Blog():

        st.markdown("""
                                    <h1 style="color:#c30452;">Blog</h1>

                                     """, unsafe_allow_html=True)

        # Define two columns with adjustable weights
        col1, col2 = st.columns([1, 1])  # You can change the ratio to adjust the widths


        # Carousel HTML template
        def generate_carousel_html(base64_images):
            return f"""
                <div class="carousel-container" style="width: 90%; margin: auto; position: relative;">
                    <div class="carousel" style="width: 100%; overflow: hidden; position: relative;">
                        {"".join([f'<img src="{img}" alt="Carousel Image" style="width: 100%; display: none;">' for img in base64_images])}
                    </div>
                    <script>
                        let currentIndex = 0;
                        const images = document.querySelectorAll('.carousel img');
                        const totalImages = images.length;

                        function showNextImage() {{
                            images[currentIndex].style.display = 'none';
                            currentIndex = (currentIndex + 1) % totalImages;
                            images[currentIndex].style.display = 'block';
                        }}

                        setInterval(showNextImage, 3000);
                        images.forEach((img, index) => {{
                            img.style.display = index === 0 ? 'block' : 'none';
                        }});
                    </script>
                </div>
            """

        def get_base64_image(image_url):
            try:
                response = requests.get(image_url)
                if response.status_code == 200:
                    return base64.b64encode(response.content).decode()
            except Exception as e:
                print(f"Error fetching image: {e}")
            return None

        def get_blog_preview(url):
            try:
                response = requests.get(url)
                soup = BeautifulSoup(response.content, 'html.parser')

                # Meta description tag'ini çekiyoruz
                meta_description = soup.find('meta', attrs={'name': 'twitter:description'})
                summary = meta_description['content'] if meta_description else "Özet bulunamadı"

                # Meta image tag'ini çekiyoruz
                meta_image = soup.find('meta', attrs={'name': 'twitter:image'})
                image_url = meta_image['content'] if meta_image else None

                return summary, image_url
            except Exception as e:
                print(f"Error fetching blog content: {e}")
            return None, None

        def main():

            # Blog yazılarının URL'leri
            blog_urls = [
                "https://www.slimmingworld.co.uk/blog/slimming-world-kitchen/",
                "https://www.slimmingworld.co.uk/blog/slimming-world-five-active-habits/",
                "https://www.slimmingworld.co.uk/blog/eat-slimming-world-summer-recipes/",
                "https://www.slimmingworld.co.uk/blog/30-summer-holiday-activity-ideas/",
                "https://www.slimmingworld.co.uk/blog/inspire-mens-weight-loss-transformations/",
            ]

            for url in blog_urls:
                summary, image_url = get_blog_preview(url)

                if image_url:
                    image_base64 = get_base64_image(image_url)
                    if image_base64:
                        st.markdown(f"""
                                <div style="text-align: center; margin-bottom: 20px;">
                                    <img src="data:image/png;base64,{image_base64}" alt="Blog Image" style="width:125%; max-width:600px;">
                                </div>
                            """, unsafe_allow_html=True)

                if summary:
                    st.markdown(f"""
                            <div style="text-align: center; margin-bottom: 20px;">
                                <p>{summary}</p>
                            </div>
                        """, unsafe_allow_html=True)

                    # Ortalanmış bağlantıyı ekleyin
                    st.markdown(f"""
                           <div style="text-align: center; margin-bottom: 20px;">
                               <a href="{url}" target="_blank" style="text-decoration: none; color: #007BFF;">Read more</a>
                           </div>
                       """, unsafe_allow_html=True)

        with col1:
            #st.header("Column 1")
            #st.write("This is the first column.")
            base64_images_col1 = [f"data:image/png;base64,{image_to_base64(path)}" for path in image_paths2]
            carousel_html_col1 = generate_carousel_html(base64_images_col1)
            st.components.v1.html(carousel_html_col1, height=600, scrolling=False)

        with col2:
            #st.header("Column 2")
            #st.write("This is the second column.")
            base64_images_col2 = [f"data:image/png;base64,{image_to_base64(path)}" for path in image_paths3]
            carousel_html_col2 = generate_carousel_html(base64_images_col2)
            st.components.v1.html(carousel_html_col2, height=600, scrolling=False)

        if __name__ == "__main__":
            main()


    def YouTube():
        def search_youtube_videos(api_key, query, max_results=10):
            youtube = build('youtube', 'v3', developerKey=api_key)
            search_response = youtube.search().list(
                q=query,
                part='id,snippet',
                maxResults=max_results
            ).execute()
            videos = []
            for search_result in search_response.get('items', []):
                if search_result['id']['kind'] == 'youtube#video':
                    video_url = f"https://www.youtube.com/watch?v={search_result['id']['videoId']}"
                    thumbnails = search_result['snippet']['thumbnails']
                    thumbnail_url = \
                        thumbnails.get('maxres',
                                       thumbnails.get('high', thumbnails.get('medium', thumbnails['default'])))[
                            'url']
                    videos.append({
                        'title': search_result['snippet']['title'],
                        'videoId': search_result['id']['videoId'],
                        'description': search_result['snippet']['description'],
                        'thumbnail': thumbnail_url,
                        'channelTitle': search_result['snippet']['channelTitle'],
                        'publishTime': search_result['snippet']['publishTime'],
                        'videoUrl': video_url
                    })
            return videos

        # Apply custom CSS for width adjustment
        st.markdown("""
            <style>
            .stTextInput div[data-baseweb="input"] > input {
                width: 600px !important;
            }
            .stButton button {
                width: 400px !important;
            }
            </style>
            """, unsafe_allow_html=True)

        #st.title("YouTube Video Search")
        st.markdown("""
                                            <h1 style="color:#c30452;">YouTube Video Search</h1>

                                             """, unsafe_allow_html=True)
        query = st.text_input("Enter keyword(s) to search for YouTube videos", "")
        search_button = st.button("Search on YouTube", key='search_button', help='Click to search with your input',
                                  use_container_width=True, type='primary')
        recommend_button = st.button("Recommend to me", key='recommend_button', help='Click to get a recommended query',
                                     use_container_width=True, type='secondary')

        if search_button and query:
            videos = search_youtube_videos(api_key, query, max_results=10)
        elif recommend_button:
            recommended_query = random.choice([
                'Weight Loss Tutorial',
                'Healthy Eating Basics',
                'Workouts for Overweights',
                'Mindfulness for Weight Control',
                'Healthy Simple Recipes',
                'What is Obesity Surgery',
                'What is Stomach Reduction',
                'Gastric Bypass',
                'Gastric Balloon'
            ])
            videos = search_youtube_videos(api_key, recommended_query, max_results=10)
            st.write(f"Recommended query: {recommended_query}")

        if 'videos' in locals():
            cols = st.columns(2)

            for i, video in enumerate(videos):
                publish_time = video['publishTime']
                parsed_time = datetime.strptime(publish_time, "%Y-%m-%dT%H:%M:%SZ")

                # Format the datetime to your desired format
                formatted_time = parsed_time.strftime("%Y-%m-%d %H:%M")
                with cols[i % 2]:
                    st.image(video['thumbnail'], use_column_width=True)
                    st.markdown(f"**[{video['title']}]({video['videoUrl']})**")
                    st.write(f"Channel: {video['channelTitle']}")
                    st.write(f"Published on: {formatted_time}")
                    st.write(video['description'])

    def Help():
        import streamlit as st
        import smtplib
        from email.message import EmailMessage

        from dotenv import load_dotenv
        import os

        # .env dosyasını yükleyin
        load_dotenv()

        password = os.getenv('EMAIL_PASSWORD')

        # E-posta göndermek için bir fonksiyon
        def send_email(name, email, question):
            msg = EmailMessage()
            msg.set_content(f"Name: {name}\nEmail: {email}\nQuestion: {question}")
            msg['Subject'] = 'New Help Request'
            msg['From'] = email  # Gönderen e-posta adresi kullanıcıdan alınır
            msg['To'] = 'weightwisehelp24@gmail.com'  # Alıcı e-posta adresi

            # SMTP sunucu ayarları (örnek olarak Gmail kullanılmıştır)
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login('weightwisehelp24@gmail.com', password)  # E-posta ve şifre
                server.send_message(msg)

        # Streamlit başlığı
        st.markdown("""
            <h1 style="color:#c30452;">Help</h1>
        """, unsafe_allow_html=True)

        st.markdown("""
            <h2 style="color:#c30452;">Submit Your Question</h2>
        """, unsafe_allow_html=True)

        # Kullanıcıdan bilgi alma
        name = st.text_input('Your Name:')
        email = st.text_input('Your E-mail:')
        question = st.text_area('Your Question (max 150 characters):', max_chars=150)

        # "Send" butonuna basıldığında e-postayı gönder
        if st.button('Send'):
            if name and email and question:
                try:
                    send_email(name, email, question)
                    st.success('Your question has been sent successfully!')
                except Exception as e:
                    st.error(f'An error occurred: {e}')
            else:
                st.warning('Please fill in all fields.')



        import streamlit.components.v1 as components
        import base64

        audio_file = open(
            'C:/Users/Tuba/PycharmProjects/pythonProject_Miuul/my_streamlit_frontend/data/Earth_Wind_Fire_September.mp3',
            'rb')
        audio_bytes = audio_file.read()
        audio_data = base64.b64encode(audio_bytes).decode()

        html_code = f"""
            <div style="display: flex; justify-content: center; margin-top: 20px;">
                <audio id="audio" src="data:audio/mp3;base64,{audio_data}" type="audio/mp3"></audio>
                <button id="playButton" style="background-color: #e56d8e; color: #ffffff; padding: 16px 36px; text-align: center; font-size: 20px; margin: 4px 2px; cursor: pointer; border-radius: 30px;">
                    🎉 Let's have some fun!!!
                </button>
            </div>
            <canvas id="confettiCanvas" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; pointer-events: none;"></canvas>
            <script src="https://cdn.jsdelivr.net/npm/canvas-confetti@1.4.0/dist/confetti.browser.min.js"></script>
            <script>
                var audio = document.getElementById('audio');
                var button = document.getElementById('playButton');
                var confettiCanvas = document.getElementById('confettiCanvas');
                var confettiInstance = null;

                button.addEventListener('click', function() {{
                    if (audio.paused) {{
                        audio.play();
                        button.innerHTML = '😶‍🌫️ Enough for today, thx :)';
                        button.style.backgroundColor = '#f44336';
                        confettiInstance = confetti.create(confettiCanvas, {{
                            resize: true,
                            useWorker: true
                        }});
                        confettiInstance({{
                            particleCount: 200,
                            spread: 160,
                            origin: {{ y: 0.5, x: 0.5 }} // Ortadan yayılmasını sağlar
                        }});
                        confettiLoop();
                    }} else {{
                        audio.pause();
                        button.innerHTML = '🎉 Let\\'s have some fun!!!';
                        button.style.backgroundColor = '#e56d8e';
                        if (confettiInstance) {{
                            confettiInstance.reset();
                        }}
                    }}
                }});

                function confettiLoop() {{
                    if (!audio.paused) {{
                        confettiInstance({{
                            particleCount: 5,
                            spread: 160,
                            origin: {{ y: 0.5, x: 0.5 }} // Ortadan yayılmasını sağlar
                        }});
                        requestAnimationFrame(confettiLoop);
                    }}
                }}
            </script>
            """
        components.html(html_code)

        # Resim dosyasını yükleme ve gösterme
        image_path_logo = 'Weight_Wise_Logo.png'

        image_path_logo = os.path.join(current_dir, 'data', image_path_logo)

        image_base64_logo = image_to_base64(image_path_logo)

        # Streamlit uygulaması
        st.markdown(f"""
                              <div style="text-align: center;">
                                  <img src="data:image/png;base64,{image_base64_logo}" alt="Weight Wise Logo" style="width:20%;"/>
                                  <p></p>
                              </div>
                                              """, unsafe_allow_html=True)

        st.markdown("""
                        <h3 style="color:#c30452;text-align: center;">Please connect with us🐝</h3>
                        <ul style="font-size:15px; color:#c30452;text-align: center;">
                                <ul style="font-size:20px; color:#c30452;">
                                <b>E-mail : weightwisehelp24@gmail.com</b>
                            </ul>
                                """, unsafe_allow_html=True)


        st.markdown('</div>', unsafe_allow_html=True)


    # Create a top navigation menu with balanced distribution
    selected = option_menu(
        menu_title=None,
        options=["Home", "Insights", "Predict", "YouTube", "Blog", "Help"],
        icons=["house", "file", "file", "file", "file", "file"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"background-color": "#e36285", "padding": "45px"},  # Remove default padding
            "icon": {"color": "#d3e785", "font-size": "20px"},
            "nav-link": {"color": "#d3e785", "font-size": "20px"},  # Adjust font size
            "nav-link-active": {"background-color": "#c30452", "color": "#fa8072"}  # Style for active link
        }
    )

    # Display the carousel and selected page's content
    if selected == "Home":
        display_carousel(image_paths1)
        home_page()
    elif selected == "Insights":
        Insights()
    elif selected == "Predict":
        Prediction()
    elif selected == "YouTube":
        YouTube()
    # elif selected == "Page 4":
    #     page_4()
    elif selected == "Blog":
        Blog()
    elif selected == "Help":
        Help()


if __name__ == "__main__":
    main()



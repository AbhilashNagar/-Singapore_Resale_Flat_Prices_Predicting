import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import statistics
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV

# -------------------------------This is the configuration page for our Streamlit Application---------------------------
st.set_page_config(
    page_title="Singapore Resale Flat Prices Prediction",
    page_icon="🏨",
    layout="wide"
)
# -------------------------------This is the sidebar in a Streamlit application, helps in navigation--------------------
with st.sidebar:
    selected = option_menu("Main Menu", ["About Project", "Predictions"],
                           icons=["house", "gear"],
                           styles={"nav-link": {"font": "sans serif", "font-size": "20px", "text-align": "centre"},
                                   "nav-link-selected": {"font": "sans serif", "background-color": "#0072b1"},
                                   "icon": {"font-size": "20px"}
                                   }
                           )

# -----------------------------------------------About Project Section--------------------------------------------------
if selected == "About Project":
    st.markdown("# :blue[Singapore Resale Flat Prices Prediction]")
    st.markdown('<div style="height: 50px;"></div>', unsafe_allow_html=True)
    st.markdown("### :blue[Technologies :] Python, Pandas, Numpy, Scikit-Learn, Streamlit, Python scripting, "
                "Machine Learning, Data Preprocessing, Visualization, EDA, Model Building, Data Wrangling, "
                "Model Deployment")
    st.markdown("### :blue[Overview :] This project aims to construct a machine learning model and implement "
                "it as a user-friendly online application in order to provide accurate predictions about the "
                "resale values of apartments in Singapore. This prediction model will be based on past transactions "
                "involving resale flats, and its goal is to aid both future buyers and sellers in evaluating the "
                "worth of a flat after it has been previously resold. Resale prices are influenced by a wide variety "
                "of criteria, including location, the kind of apartment, the total square footage, and the length "
                "of the lease. The provision of customers with an expected resale price based on these criteria is "
                "one of the ways in which a predictive model may assist in the overcoming of these obstacles.")
    st.markdown("### :blue[Domain :] Real Estate")

# ------------------------------------------------Predictions Section---------------------------------------------------
if selected == "Predictions":
    st.markdown("# :blue[Predicting Results based on Trained Models]")
    st.markdown("### :orange[Predicting Resale Price (Regression Task) (Accuracy: 87%)]")

    try:
        with st.form("form1"):

            # -----New Data inputs from the user for predicting the resale price-----
            floor_area_sqm = st.number_input('Floor Area (Per Square Meter)', min_value=1.0, max_value=500.0)
            lease_commence_date = st.number_input('Lease Commence Date')
            storey_range = st.text_input("Storey Range (Format: 'Value1' TO 'Value2')")

            # -----Submit Button for PREDICT RESALE PRICE-----
            submit_button = st.form_submit_button(label="PREDICT RESALE PRICE")

            if submit_button is not None:
                with open(r"model.pkl", 'rb') as file:
                    loaded_model = pickle.load(file)
                with open(r'scaler.pkl', 'rb') as f:
                    scaler_loaded = pickle.load(f)

                # -----Calculating lease_remain_years using lease_commence_date-----
                lease_remain_years = 99 - (2023 - lease_commence_date)

                # -----Calculating median of storey_range to make our calculations quite comfortable-----
                split_list = storey_range.split(' TO ')
                float_list = [float(i) for i in split_list]
                storey_median = statistics.median(float_list)

                # -----Sending the user enter values for prediction to our model-----
                new_sample = np.array(
                    [[np.log(floor_area_sqm), lease_remain_years, np.log(storey_median)]])
                new_sample = scaler_loaded.transform(new_sample[:, :5])
                new_pred = loaded_model.predict(new_sample)[0]
                st.write('## :green[Predicted resale price:] ', np.exp(new_pred))

    except Exception as e:
        st.write("Enter the above values to get the predicted resale price of the flat")
import pandas as pd
import streamlit as st
import machine_learning as ml
import feature_extraction as fe
from bs4 import BeautifulSoup
import requests as re
import matplotlib.pyplot as plt
import numpy as np


st.title("MALICIOUS URL DETECTION BY PHISHING IDENTIFICATION")
st.write(
    'This ML-based app is developed by Umadevi Narri 19011P0413 ECE IDP.'
    'Objective of the app is detecting phishing websites  using both url based and content based approach. '
    'You can see the details of approach, data set, and feature set, click on the **PROJECT DETAILS** section in the side menu.')

# Sidebar Navigation
st.sidebar.title("MENU")
selected_option = st.sidebar.selectbox("Select Option", ["Project Details", "Dataset","Result","Try Now"])
if selected_option=="Project Details":
    with st.expander('Approach'):
        st.header("APPROACH")
        st.write('The approach combines both URL-based and content-based features for phishing website detection. Content-based features are extracted from the HTML of websites, while URL-based features are obtained from the Links. A _supervised learning_ approach is employed for classification using various machine learning models.')
    with st.expander('Aim'):
        st.subheader("AIM")
        st.write(
            'To implement a machine learning approach capable of effectively differentiating between legitimate websites and fraudulent/phishing websites, thereby enhancing cybersecurity by preventing users from falling victim to phishing attacks and ensuring a safer online experience.')

    with st.expander("Objectives"):
        st.subheader('OBJECTIVES')
        st.write('1. Conduct a comprehensive study on Phishing detection methods.')
        st.write('2. Acquire and preprocess an appropriate dataset relevant to the problem statement.')
        st.write(
            '3. Identify and select suitable machine learning techniques to develop a solution for phishing detection.')
        st.write('4. Evaluate the performance of machine learning algorithm.')
        st.write('5. Deploying a machine learning web app.')

elif selected_option=="Dataset":
    st.header("DATA SET")
    st.write('For my project,'
                 'I created a custom dataset by collecting data from various sources. I defined features based on a combination of existing literature and manual analysis of phishing website characteristics. I used the `requests` library to collect website data, and the `BeautifulSoup` module to parse and extract relevant features from the HTML content.')
    st.write(
        'I used [_"phishtank.org"_](https://www.phishtank.com/) & [_"tranco-list.eu"_](https://tranco-list.eu/) as data sources.')
    st.write('Totally 26584 websites ==> **_16060_ legitimate** websites | **_10524_ phishing** websites')
    st.write('Data set was created in October 2023.')

    # ----- FOR THE PIE CHART ----- #
    labels = 'phishing', 'legitimate'
    phishing_rate = int(ml.phishing_df.shape[0] / (ml.phishing_df.shape[0] + ml.legitimate_df.shape[0]) * 100)
    legitimate_rate = 100 - phishing_rate
    sizes = [phishing_rate, legitimate_rate]
    explode = (0.1, 0)
    fig, ax = plt.subplots()
    ax.pie(sizes, explode=explode, labels=labels, shadow=True, startangle=90, autopct='%1.1f%%')
    ax.axis('equal')
    st.pyplot(fig)
    # ----- !!!!! ----- #
    st.write('Features + URL + Label ==> Dataframe')
    st.markdown('label is 1 for phishing, 0 for legitimate')
    number = st.slider("Select row number to display", 0, 100)
    newdf=pd.read_csv("url_content_combined_dataset.csv")
    st.dataframe(newdf.head(number))


    # download button as a csv

    @st.cache_data
    def convert_df(df):
        # IMPORTANT: Cache the conversion to prevent computation on every rerun
        return df.to_csv().encode('utf-8')


    csv = convert_df(newdf)

    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='phishing_legitimate_structured_data.csv',
        mime='text/csv',
    )

    st.subheader("FEATURES")
    # Define URL-based and content-based features
    url_features = ['use_of_ip', 'abnormal_url', 'google_index', 'count.', 'count-www', 'count@', 'count_dir',
                    'count_embed_domian', 'short_url', 'count-https', 'count-http', 'count%', 'count?', 'count-',
                    'count=', 'url_length', 'hostname_length', 'sus_url', 'count-digits', 'count-letters', 'fd_length',
                    'URL', 'label']

    content_features = ['has_title', 'has_input', 'has_button', 'has_image', 'has_submit', 'has_link', 'has_password',
                        'has_email_input', 'has_hidden_element', 'has_audio', 'has_video', 'number_of_inputs',
                        'number_of_buttons', 'number_of_images', 'number_of_option', 'number_of_list', 'number_of_th',
                        'number_of_tr', 'number_of_href', 'number_of_paragraph', 'number_of_script', 'length_of_title',
                        'has_h1', 'has_h2', 'has_h3', 'length_of_text', 'number_of_clickable_button', 'number_of_a',
                        'number_of_img', 'number_of_div', 'number_of_figure', 'has_footer', 'has_form', 'has_text_area',
                        'has_iframe', 'has_text_input', 'number_of_meta', 'has_nav', 'has_object', 'has_picture',
                        'number_of_sources', 'number_of_span', 'number_of_table']

    # Display the corresponding feature list when a button is clicked
    if st.button("URL-based Features:"):
        st.write(url_features)

    if st.button("Content-based Features:"):
        st.write(content_features)

elif selected_option=="Result":
    st.header("RESULTS")
    st.write('I used 5 different ML classifiers of scikit-learn and tested them implementing k-fold cross validation.')
    st.write('Firstly obtained their confusion matrices, then calculated their accuracies.')
    st.write()
    models = ['RF', 'DT', 'Ada', 'NB', 'XGB']
    new_accuracies = [89, 80, 90, 61, 93]
    colors = ['#cfb148', '#cc4949', '#8c679e', '#5d8f8e', '#632424']

    # Create a single bar plot for new accuracies
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, new_accuracies, color=colors)
    ax.set_xlabel('Models')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy of Different Models')
    ax.set_ylim(0, 100)  # Set the y-axis limit to ensure the accuracy values are displayed properly


    # Function to add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.annotate('{}'.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')


    # Add value labels on top of bars
    add_labels(bars)

    # Show the plot within the Streamlit app
    st.pyplot(fig)

    st.write("RF --> Random Forest")
    st.write("DT --> Decision Tree")
    st.write("Ada --> AdaBoost")
    st.write("NB --> Naive Bayes")
    st.write("XGB --> XGBoost")

elif selected_option=="Try Now":
    st.write('EXAMPLE PHISHING URLs:')
    st.write('_https://rtyu38.godaddysites.com/_')
    st.write('_https://karafuru.invite-mint.com/_')
    st.write('_https://defi-ned.top/h5/#/_')
    st.write('To access more phishing URLs')
    st.write('_[OpenPhish](https://openphish.com/index.html)_')
    st.caption('REMEMBER, PHISHING WEB PAGES HAVE SHORT LIFECYCLE!')

    choice = st.selectbox("Please select your machine learning model",
                          ['Decision Tree', 'Random Forest', 'Naive Bayes', 'AdaBoost', 'XGBOOST'])

    model = ml.xgb_model

    if choice == 'Naive Bayes':
        model = ml.nb_model
        st.write('NB model is selected!')
    elif choice == 'Decision Tree':
        model = ml.dt_model
        st.write('DT model is selected!')
    elif choice == 'Random Forest':
        model = ml.rf_model
        st.write('RF model is selected!')
    elif choice == 'AdaBoost':
        model = ml.ada_model
        st.write('ADA model is selected!')
    else:
        model = ml.xgb_model
        st.write('XGB model is selected!')

    url = st.text_input('Enter the URL')
    # check the url is valid or not
    if st.button('Check!'):
        try:
            response = re.get(url, verify=False, timeout=4)
            if response.status_code != 200:
                st.write(". HTTP connection was not successful for the URL: ", url)
            else:
                soup = BeautifulSoup(response.content, "html.parser")
                url_features = fe.create_vector_url(url)
                content_features = fe.create_vector_soup(soup)
                vector = url_features + content_features
                vector = np.array(vector).reshape(1, -1)
                result = model.predict(vector)
                if result[0] == 0:
                    st.success("This web page seems a legitimate!")
                    st.balloons()
                else:
                    st.warning("Attention! This web page is a potential PHISHING!")
                    st.snow()

        except re.exceptions.RequestException as e:
            st.error("An error occurred while trying to access the URL. Please try again later.")
            st.error(str(e))


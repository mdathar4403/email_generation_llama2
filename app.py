import streamlit as st
import nltk  # Natural Language Toolkit for text processing
from nltk.corpus import stopwords 
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

import pathlib
import textwrap

import google.generativeai as genai

#used to store your API key
# from google.colab import userdata

from IPython.display import display
from IPython.display import Markdown

from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
# TheBloke/Llama-2-7B-Chat-GGML
import re
# Set the API key
import os
os.environ['GOOGLE_API_KEY']='AIzaSyBKKVhRIDqVi8QStwrMaHAyFmPle0JLWQo'
genai.configure(api_key = os.environ['GOOGLE_API_KEY'])
model = genai.GenerativeModel('gemini-pro')

# def to_markdown(text):
#   text = text.replace("*", "*")
#   return Markdown(textwrap.indent(text, '>', predicate=lambda _:True))

def getLLamaresponse(your_name, input_text,from_station, to_station):

    llm=CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    ## Prompt Template

    template='''Write a sorry email using standard mail format to the customer {your_name} on behalf of railway authority regarding the issue {input_text} faced by the customer while travelling
        from station {from_station} to station {to_station} helping the customer with the issue within 300 words. In last Yours faithfully, Railway Authority'''
    
    prompt=PromptTemplate(input_variables=["your_name","input_text","from_station","to_station"],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm.invoke(prompt.format(your_name=your_name,input_text=input_text,from_station=from_station,to_station=to_station))
    text = re.sub(r"(?<!\w\.)\.(?=\s)", "\n", response)
    # print(response)
    return text

# Gemini AI
def gemini_ai(your_name, input_text, from_station, to_station):
#   """Generates a sorry email for train delay using Gemini AI."""
  response = model.generate_content(f"Write a sorry email to customer {your_name} for the issue {input_text} from {from_station} to {to_station} on behalf of railway authority.")
  return response.text  # Return the plain text content

def preprocess_text(text):
    if text=="":
        return "Please enter the case desciption"
    # # Method 1: Using Spacy
    doc = nlp(text)
    # keywords = [(w.text, w.pos_) for w in doc]
    keywords = [token.text for token in doc if token.pos_ in ["VERB","ADJ","NOUN"]]
    return keywords

    #Method 2: Brute Force way
    # Download stopwords (first run only)
    # nltk.download('stopwords')
    # text = text.lower()  # Lowercase
    # text = ''.join([c for c in text if c.isalpha() or c.isspace()])  # Remove punctuation
    # tokens = text.split()  # Tokenize
    # stop_words = set(stopwords.words('english'))  # Load English stopwords
    # filtered_words = [w for w in tokens if w not in stop_words]  # Remove stopwords (optional)
    # return filtered_words

def calculate_score(text):
    if text=="":
        return "Please enter the case desciption"
    result=sia.polarity_scores(text)
    score = result['compound']
    
    if score>0:
        return "It is Positive 😊"
    elif score<0:
        return "It is Negative 😒"
    else:
        return "It is Neutral 😑"
    
st.set_page_config(page_title="We are here to help you !! 🤖",
                    page_icon='🤖',
                    layout='centered',
                    initial_sidebar_state='collapsed')

# Define a two-column layout
col1, col2 = st.columns([2, 3])

# Column 1: Display the image
with col1:
    st.image("https://logowik.com/content/uploads/images/indian-railways3115.jpg/", width=200, caption="Railway Authority")

# Column 2: Display the header text
with col2:
    st.header("Get Help on the Right Track (Fast!) 🤖")


# Create three tabs
tab1, tab2, tab3 = st.tabs(["Email Generation", "Sentiment Score", "Goodwill Voucher"])

# Content for Home Tab
with tab1:
    st.subheader("Get the Help You Deserve: Let's Get Started")

    your_name=st.text_input("Enter your Name:")
    input_text=st.text_area("Enter your issue here:")
    from_station = st.text_input("Enter the from station:")
    to_station = st.text_input("Enter the to station:")

    option = ["Llama-2 ♾️", "Gemini ✨"]
    selected_option = st.selectbox("Select a option:", option)
    
    
    if selected_option=="Llama-2 ♾️":
        submit=st.button("Generate")
        if submit:
            with st.expander("Open Mail"):
                st.write(getLLamaresponse(your_name, input_text,from_station, to_station))
    else:
        submit=st.button("Generate")
        if submit:
            with st.expander("Open Mail"):
                st.write(gemini_ai(your_name, input_text,from_station, to_station))

with tab2:
    st.subheader("Welcome to Keywords/Sentiment score section")
    
    enter_issue = st.text_area("Enter the issue here:")
    keywords = ["Print Keywords", "Calculate sentiment score"]
    selected_keyword = st.selectbox("Select a option:", keywords)
    
    if selected_keyword=='Print Keywords':
        submit= st.button('Print Keywords')
        if submit:
            with st.expander("Show Keywords"):
                st.write(preprocess_text(enter_issue))
    else:
        submit=st.button("Calculate Score")
        if submit:
            st.write(calculate_score(enter_issue))

with tab3:
    st.subheader("Welcome to Goodwill Voucher section")
    
    case_desc = st.text_area("Enter the case Description here:")
    severity = st.text_input("Enter the Severity(1-10):")
    interf_travel_plan = st.text_input("Enter the Interference travel plan(1-10):")
    justi_complaint = st.text_input("Enter the Justification of complaint(1-10):")
    duration = st.text_input("Enter the Customer History(1-10):")
    
    submit=st.button("Predict Voucher")
    if submit:
        st.write()
    
    
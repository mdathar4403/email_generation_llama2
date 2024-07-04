import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
# TheBloke/Llama-2-7B-Chat-GGML

def getLLamaresponse(your_name, input_text,from_station, to_station):

    llm=CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                      model_type='llama',
                      config={'max_new_tokens':256,
                              'temperature':0.01})
    
    ## Prompt Template

    template="""
        Write a sorry email to the customer {your_name} on behalf of railway authority regarding the issue {input_text} faced by the customer while travelling
        from station {from_station} to station {to_station} helping the customer with the issue within 300 words. In last Yours faithfully, Railway Authority
            """
    
    prompt=PromptTemplate(input_variables=["your_name","input_text","from_station","to_station"],
                          template=template)
    
    ## Generate the ressponse from the LLama 2 model
    response=llm(prompt.format(your_name=your_name,input_text=input_text,from_station=from_station,to_station=to_station))
    print(response)
    return response




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
    st.header("We are here to help you !! 🤖")


# Create three tabs
tab1, tab2, tab3 = st.tabs(["Email Generation", "Sentiment Score", "Goodwill Voucher"])

# Content for Home Tab
with tab1:
    st.subheader("Please enter your details below and we will generate the response for you")

    your_name=st.text_input("Enter your Name:")
    input_text=st.text_input("Enter your issue here:")
    from_station = st.text_input("Enter the from station:")
    to_station = st.text_input("Enter the to station:")

    submit=st.button("Generate")

## Final response
if submit:
    st.write(getLLamaresponse(your_name, input_text,from_station, to_station))
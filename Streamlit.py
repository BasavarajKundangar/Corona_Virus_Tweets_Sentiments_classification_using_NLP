import streamlit as st
from PIL import Image
image = Image.open('pic.jpg')
image2 = Image.open('pic2.jpg')
#app = Flask(__name__)
import requests
import json


def main():
    
    html_temp = """
    <div style="background-color:DodgerBlue;padding:10px">
    <h2 style="color:white;text-align:center;">Twitter Sentimental Analysis ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.image(image,use_column_width=True, caption='Sentimental Analysis',width=400)
    Text = st.text_input("Enter your Text here")
    result=""
    obj=""    
    if st.button("Predict"):
        txt={"text":Text}
        url = 'http://127.0.0.1:8000/predict'
        x = requests.post(url, json = txt)
        obj=x.json()
        result = obj[0] 
    st.success(result)
    st.image(image2,use_column_width=True,width=800)

if __name__== '__main__':
    main()

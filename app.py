import streamlit as st
import textmodel
import imagecaption

st.title("Toxic Content Classification")


# Cache the function so it only runs on the first load
@st.cache_resource
def get_cached_components():
    return textmodel.LoadModel()

components = get_cached_components()

user_input = st.text_area("Enter text for classification:")

b1 = st.button("Predict")

if b1:
    if user_input.strip() != "":
        # 3. Pass the text and the cached components to your predict function
        result = textmodel.predict(user_input, components)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text first.")



st.text_area("Or upload a picture:")
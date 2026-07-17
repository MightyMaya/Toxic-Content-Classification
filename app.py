import streamlit as st
import textmodel
import imagecaption

st.title("Toxic Content Classification")


# Cache the functions so they only run on the first load
@st.cache_resource
def get_cached_components():
    return textmodel.LoadModel()

@st.cache_resource
def get_cached_components2():
    return imagecaption.LoadBlip()

text_components = get_cached_components()


user_input = st.text_area("Enter text for classification:")

b1 = st.button("Predict")

if b1:
    if user_input.strip() != "":
        # 3. Pass the text and the cached components to your predict function
        result = textmodel.predict(user_input, text_components)
        st.success(f"Prediction: {result}")
    else:
        st.warning("Please enter some text first.")



# Create the file uploader widget
uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

b2 = st.button("Predict ")

result = None

if b2:
    #load image caption model
    image_components = get_cached_components2()
    
    if uploaded_file is not None:
        #Generate a caption
        result = imagecaption.predict(uploaded_file, image_components)
        #Display the image on the app with the generated caption
        st.image(uploaded_file, caption=f"Caption: {result}", width="stretch")
          
        text_result = textmodel.predict(result, text_components)
        st.success(f"Prediction: {text_result}")
    else:
        st.warning("Please upload an image first.")    

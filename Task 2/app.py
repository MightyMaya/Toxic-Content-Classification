import streamlit as st
import database
import textmodel
import imagecaption


st.title("Toxic Content Classification")
#instantiate 2 columns
c1, c2 = st.columns(2)

# Cache the functions so they only run on the first load
@st.cache_resource
def get_cached_components():
    return textmodel.LoadModel()

@st.cache_resource
def get_cached_components2():
    return imagecaption.LoadBlip()

#intilizaliz the database
st.sidebar.header("Database Entries")
database.initialize_db()
text_components = get_cached_components()


# allow user to input text or an image in the main part of webpage
with st.form(key="classification_form", clear_on_submit=False):
    #instantiate 2 columns
    c1, c2 = st.columns(2)
    
    #column 1 for text input
    with c1:
        user_input = st.text_area("Enter text for classification:")
        
    #column 2 for image input
    with c2:
        uploaded_file = st.file_uploader("Or upload an image...", type=["jpg", "jpeg", "png"])

    
    submit_btn = st.form_submit_button(label="Submit")

#handle execution AFTER the form is submitted
if submit_btn:
    #track if any action was performed
    processed = False
    
    #process the text
    if user_input.strip() != "":
        text_result = textmodel.predict(user_input, text_components)
        c1.success(f"Text Prediction: {text_result}")
        #insert the text and its classification into the database
        entry = {"text": user_input,
                 "category": text_result}
        database.insert(entry)

        #show database in real time
        entries = database.get_entries()
        for row in entries:
            st.sidebar.write(f"{row[0]}. {row[1]}, {row[2]}")
        processed = True
        
    #process the image
    if uploaded_file is not None:
        #only load the image captioning model if there was image entered by user
        image_components = get_cached_components2()
        caption_result = imagecaption.predict(uploaded_file, image_components)
        #show the image with its caption below
        c2.image(uploaded_file, caption=f"Generated Caption: {caption_result}", use_container_width=True)
        img_text_result = textmodel.predict(caption_result, text_components)
        c2.success(f"Image Prediction: {img_text_result}")
        #insert the caption and its classification into the database
        entry = {"text": caption_result,
                 "category": img_text_result}
        database.insert(entry)

        #show database in real time
        entries = database.get_entries()
        for row in entries:
            st.sidebar.write(f"{row[0]}. {row[1]}, {row[2]}")
        processed = True
        
    #if the button was clicked but nothing was filled out
    if not processed:
        st.warning("Please enter text or upload an image.")

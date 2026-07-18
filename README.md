# Text Classification Streamlit App

This repository contains a Streamlit web application that uses a fine-tuned DistillBert (with LoRA) from hugging face to classify text into one of 9 categories:
 * Safe
 * Violent Crimes 
 * Elections 
 * Sex-Related Crimes
 * unsafe
 * Non-Violent Crimes 
 * Child Sexual Exploitation
 * Unknown S-Type
 * Suicide & Self-Harm

## Prerequisites

Before running the application, make sure you have **Python 3.8 or higher** installed. Also, you need to download the weights for the image captioning model from *[link](https://drive.google.com/drive/folders/1_isbZ76Xy6uc08WzAlXH6oENcikKNj4b?usp=sharing)*

## Installation

1. **Clone this repository** (or navigate to your project folder):
   ```bash
   cd path/to/your/project
   ```

2. **Create a virtual environment** (recommended to avoid dependency conflicts):
   ```bash
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate

   # On Windows
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install the required packages**:
   ```bash
   pip install streamlit torch transformers scikit-learn joblib
   ```
   *(Note: If you are using CUDA/GPU on Windows/Linux, ensure you install the correct PyTorch version from the official [PyTorch website](https://pytorch.org)).*

## Project Structure

Ensure your folder looks similar to this:
```text
├── app.py                 # Main Streamlit web application
├── textmodel.py        # Python script handling the text classification model's logic and predictions
├── label_encoder.pkl      # Saved label encoder file
├── imagecaption.py     # Python script handling the image captioning model's logic and predictions
├── database.py         # Python script handling the database logic
├── blip2_4bit        # You should put the blip2 model weigts downloaded from google drive in this folder
├── final_model       # This folder conatins the finetund model weigts for DitilBert
└──
```

## Running the App

To launch the web interface, run the following command in your terminal:

```bash
streamlit run app.py
```

Once the command runs, Streamlit will automatically open the application in your default web browser. If it doesn't open automatically, look at your terminal output and copy the local URL (usually `http://localhost:8501`).

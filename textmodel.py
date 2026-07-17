import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import PeftModel
import joblib


# load all necesaary things in 1 function
def LoadModel():
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("./final_model")
    # Load base model
    base_model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=9
    )
    # Load LoRA adapter
    model = PeftModel.from_pretrained(
        base_model,
        "./final_model"
    )
    model.to(device)
    # Load label encoder
    label_encoder = joblib.load("label_encoder.pkl")

    return {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "label_encoder": label_encoder
    }

def predict(text, components):
    tokenizer = components["tokenizer"]
    model = components["model"]
    device = components["device"]
    label_encoder = components["label_encoder"]
    
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = outputs.logits.argmax(dim=-1).item()

    return label_encoder.inverse_transform([prediction])[0]


# Load the model in app.py
#if __name__ == "__app__":
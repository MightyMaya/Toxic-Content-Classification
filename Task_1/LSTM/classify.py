import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from peft import PeftModel
import joblib

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
#model.eval()

# Load label encoder
label_encoder = joblib.load("label_encoder.pkl")

def predict(text):
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

while True:
    text = input("\nEnter text (or type 'quit' to exit): ")

    if text.lower() in ["quit", "exit", "q"]:
        print("Goodbye!")
        break

    prediction = predict(text)
    print(f"Prediction: {prediction}")
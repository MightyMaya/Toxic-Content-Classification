import os
from transformers import Blip2ForConditionalGeneration, AutoProcessor
import torch
from PIL import Image

#load the model
def LoadBlip():

    local_path = "./blip2_4bit" 
    
    processor = AutoProcessor.from_pretrained(local_path)
    model = Blip2ForConditionalGeneration.from_pretrained(
        local_path, 
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    return {
        "model": model,
        "processor": processor
    }


#convert image to RGB then predict
def predict(image, components): 
    processor = components["processor"]
    model = components["model"]

    raw_image = Image.open(image).convert("RGB")
    inputs = processor(raw_image, return_tensors="pt").to("cuda")

    out = model.generate(**inputs)
    return processor.decode(out[0], skip_special_tokens=True).strip()

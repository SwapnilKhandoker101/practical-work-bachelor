import streamlit as st
from transformers import AutoImageProcessor, CLIPForImageClassification,BlipProcessor, BlipForConditionalGeneration
import torch
from datasets import load_dataset
import numpy as np
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration,pipeline
from PIL import Image
import pandas as pd
from diffusers import StableDiffusionPipeline
import torch

st.title('Prompt Generation')

# models
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

file_uploader=st.file_uploader('Upload the image')
image_file=file_uploader.name
st.write(file_uploader.name)
if file_uploader is not None:
    image=Image.open(file_uploader)

st.image(image)

# Process the image
inputs = processor(image, return_tensors="pt")

# Generate a caption
with torch.no_grad():
    out = model.generate(**inputs)

# Decode the generated caption
caption = processor.decode(out[0], skip_special_tokens=True)
print("Generated Caption:", caption)
st.write(caption)

# Creating prompt:

df = pd.read_csv('IAPSAll.ibm', delimiter='\t')
df=df[['desc','IAPS']]

image_file_name = image_file.strip('.jpg')
image_file_name_float=float(image_file_name)

matched_row=df[df['IAPS']==image_file_name_float]
matched_name=matched_row['desc'].values[0]
matched_name_str=str(matched_name)

prompt=f'Generate a photo-realistic image of {matched_name_str} and {caption} '
st.write(prompt)




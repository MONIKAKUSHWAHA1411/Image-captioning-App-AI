# Install required packages first (run this in terminal or notebook)
# !pip install transformers torch torchvision pillow streamlit

import streamlit as st                    # For UI
from PIL import Image                     # For image loading and handling
import torch                              # Required for model inference
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load the pre-trained BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    use_safetensors=True,
    trust_remote_code=True
)


# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Streamlit UI setup
st.title("🖼️ AI Image Caption Generator")
st.write("Upload an image and let AI describe it for you!")

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Generate caption
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    st.markdown("### 📝 Generated Caption:")
    st.success(caption)

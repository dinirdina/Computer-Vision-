import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.models import ResNet18_Weights
from PIL import Image
import pandas as pd

# ---------------------------------------------------------
# Step 1: Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="CPU Image Classification (ResNet18)",
    layout="centered",
)
st.title("Computer Vision: Image Classification (CPU)")
st.write("Pre-trained **ResNet18** from torchvision (ImageNet) + Streamlit.")

# ---------------------------------------------------------
# Step 2 + Step 3: CPU only
# ---------------------------------------------------------
device = torch.device("cpu")
st.info(f"Device in use: **{device}** (CPU-only as required)")

# ---------------------------------------------------------
# Step 4: Load pre-trained ResNet18 & eval mode
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    weights = ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.eval()
    model.to(device)
    return model, weights

model, weights = load_model()

# ---------------------------------------------------------
# Step 5: Recommended transforms for the selected weights
# ---------------------------------------------------------
preprocess = weights.transforms()

# Labels (class names)
categories = weights.meta["categories"]

# ---------------------------------------------------------
# Step 6: Upload image UI
# ---------------------------------------------------------
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.warning("Please upload an image to start.")
    st.stop()

# Display image
image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="Uploaded Image", use_container_width=True)

# ---------------------------------------------------------
# Step 7: Convert to tensor + inference (no gradients)
# ---------------------------------------------------------
input_tensor = preprocess(image).unsqueeze(0).to(device)  # [1, 3, 224, 224]

with torch.no_grad():
    outputs = model(input_tensor)[0]  # logits

# ---------------------------------------------------------
# Step 8: Softmax + Top-5
# ---------------------------------------------------------
probs = F.softmax(outputs, dim=0)
top5_prob, top5_catid = torch.topk(probs, 5)

top5_labels = [categories[idx] for idx in top5_catid.tolist()]
top5_probs = top5_prob.tolist()

st.subheader("Top-5 Predictions (with probabilities)")
for label, p in zip(top5_labels, top5_probs):
    st.write(f"**{label}** — {p:.4f}")

# Table
df = pd.DataFrame({"Class": top5_labels, "Probability": top5_probs})
st.write("### Prediction Table")
st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------
# Step 9: Bar chart (Streamlit built-in)
# ---------------------------------------------------------
st.write("### Probability Bar Chart (Top-5)")
st.bar_chart(df.set_index("Class"))

# ---------------------------------------------------------
# Step 10: Testing note (you discuss this in report)
# ---------------------------------------------------------
st.caption("Tip: Test with multiple images and discuss whether predictions match the image content.")

from pyngrok import ngrok
import threading
import streamlit as st

# Open a public tunnel to the Streamlit app
def start_ngrok():
    url = ngrok.connect(addr=8501)
    print("Public URL:", url)

threading.Thread(target=start_ngrok).start()
import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import cv2
import tempfile
import os
import gdown

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Fake Content Detector",
    page_icon="🧠",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
    .title {
        font-size: 40px;
        font-weight: 700;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: gray;
        margin-bottom: 30px;
    }
    .result-box {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🧠 AI Fake Content Detector</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Detect whether images and videos are <b>AI-Generated</b> or <b>Real</b></div>",
    unsafe_allow_html=True
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Input Settings")
option = st.sidebar.radio("Choose content type", ("Image", "Video"))

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- MODEL DOWNLOAD ----------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "ai_detector.pth")
MODEL_URL = "https://drive.google.com/uc?id=1LF3uhDOCkuvz-pHY4ZJ7XgzWxM0XjoRR"

os.makedirs(MODEL_DIR, exist_ok=True)

if not os.path.exists(MODEL_PATH):
    with st.spinner("🔄 Downloading AI model (one-time setup)..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    st.success("✅ Model ready!")

# ---------------- LOAD MODEL ----------------
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- IMAGE DETECTION ----------------
if option == "Image":
    st.subheader("🖼 Upload Image")
    uploaded_image = st.file_uploader(
        "Supported formats: JPG, JPEG, PNG",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, 1).item()

        st.markdown("### 🔍 Detection Result")
        if pred == 0:
            st.error("🚨 **AI-GENERATED IMAGE**")
        else:
            st.success("✅ **REAL IMAGE**")

# ---------------- VIDEO DETECTION ----------------
if option == "Video":
    st.subheader("🎥 Upload Video")
    uploaded_video = st.file_uploader(
        "Supported formats: MP4, MOV, AVI",
        type=["mp4", "mov", "avi"]
    )

    if uploaded_video:
        st.video(uploaded_video)

        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)

        ai_count = 0
        real_count = 0
        frame_no = 0

        with st.spinner("🔍 Analyzing video frames..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_no += 1
                if frame_no % 10 != 0:
                    continue

                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = transform(img).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = model(img)
                    pred = torch.argmax(output, 1).item()

                if pred == 0:
                    ai_count += 1
                else:
                    real_count += 1

        cap.release()

        st.markdown("### 📊 Detection Summary")
        col1, col2 = st.columns(2)
        col1.metric("AI Frames", ai_count)
        col2.metric("Real Frames", real_count)

        if ai_count > real_count:
            st.error("🚨 **VIDEO IS LIKELY AI-GENERATED**")
        else:
            st.success("✅ **VIDEO IS LIKELY REAL**")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("📘 School Project | AI Fake Content Detector | Built with Streamlit & PyTorch")


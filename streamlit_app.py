import streamlit as st
import torch
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
import os

# Debugging awal

model_dir = "model"


# Judul Aplikasi
st.title("🔍 Prediksi Kelas Gaharu")
st.write("Unggah satu atau lebih gambar untuk diprediksi kelasnya.")

# Path model yang ingin kamu pakai
model_path = os.path.join(model_dir, "efficientnet_model_best_deploy_100.pth")

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        st.error(f"❌ File model tidak ditemukan: `{model_path}`")
        return None, None

    try:
        checkpoint = torch.load(model_path, map_location=torch.device("cpu"))

        if "model_state_dict" not in checkpoint or "class_names" not in checkpoint:
            st.error("❌ Struktur file model tidak valid. Harus ada `model_state_dict` dan `class_names`.")
            return None, None

        class_names = checkpoint["class_names"]
        num_classes = len(class_names)

        model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=num_classes)
        model._fc = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model._fc.in_features, num_classes)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

   
        return model, class_names

    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}")
        return None, None


# Load model
model, class_names = load_model()

# Transformasi untuk gambar input
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def predict_image(image):
    image = transform_test(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        predicted_idx = torch.argmax(probabilities).item()

    return class_names[predicted_idx], max(probabilities).item()


# Upload & prediksi
uploaded_files = st.file_uploader("📤 Unggah satu atau beberapa gambar", type=["jpg", "jpeg", "png", "webp"], accept_multiple_files=True)

if uploaded_files and model:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"📷 {uploaded_file.name}", use_container_width=True)

        pred_class, confidence = predict_image(image)
        st.success(f"✅ Prediksi: **{pred_class}** \n🎯 Confidence: {confidence:.2f}")
elif uploaded_files and not model:
    st.error("❌ Model belum berhasil dimuat, tidak bisa memproses gambar.")

import streamlit as st
from PIL import Image
from ultralytics import YOLO
from collections import Counter

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

st.title("🦠 Blood Cell Count with YOLOv8")
st.write("Upload a microscopy image. The model will detect and count different types of blood cells.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Display original image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Run YOLOv8 model
    results = model.predict(image)
    result_img = results[0].plot()
    results = results[0]

    # Get class IDs from prediction
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    class_names = results.names

    # Count occurrences of each class
    counts = Counter(class_ids)

    # Prepare dictionary with readable names
    blood_cell_count = {class_names[cls_id]: count for cls_id, count in counts.items()}

    # Display detection image
    st.image(result_img, caption="Detected Blood Cells", use_column_width=True)

    # Display count
    if blood_cell_count:
        st.subheader("🔬 Blood Cell Count:")
        for cell_type, count in blood_cell_count.items():
            st.write(f"**{cell_type}**: {count}")
    else:
        st.warning("No blood cells detected.")

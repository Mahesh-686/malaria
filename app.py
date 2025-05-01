import streamlit as st
from PIL import Image
from ultralytics import YOLO
from collections import Counter

# Load model
model = YOLO("best.pt")

st.title("🦠 Bloodcell Count with YOLOv8")
st.write("Upload a microscopy image. The model will highlight detected cells.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Inference
    results = model.predict(image)
    result_img = results[0].plot()
    results = results[0]  # Get the first result object

    # Get class IDs and names
    class_ids = results.boxes.cls.cpu().numpy().astype(int)
    class_names = results.names
    counts = Counter(class_ids)

    # Prepare output
    blood_cell_count = {}
    for cls_id, count in counts.items():
        cls_name = class_names[cls_id]
        

    # Display detection image
    st.image(result_img, caption="Detected Cells", use_column_width=True)

    # Display final sentence
    if blood_cell_count:
        st.write(f"Blood cells are {blood_cell_count}.")
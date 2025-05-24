import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from collections import OrderedDict
from torchvision import models
from fuzzy import *
from vals_for_labels import *

# Reconstruct the model
model = models.resnet50()
model.fc = torch.nn.Linear(model.fc.in_features, 101)

# Load weights
model.load_state_dict(torch.load("Food101pred.pth", map_location='cpu'))
model.eval()
    
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

advices = {
    "Top Shelf": "This is the most accessible shelf with stable temperature. It's recommended to use for ready-to-eat items, leftovers, drinks, and foods with low moisture, low to medium sensitivity and short to medium storage time.",
    "Middle Shelf": "This shelf has consistent temperature, good for dairy and foods that don't spoil too quickly. It is good for products with medium moisture, medium sensitivity and medium storage time.",
    "Lower Shelf": "This shelf is the coldest part of the fridge, ideal for raw ingredients. It's best for storing products with high moisture, medium sensitivity and longer storage time.",
    "Crisper Drawer (Left)": "It's a high-humidity drawer. The conditions are good for products with high moisture, medium to high sensitivity and short to medium storage time - usually leafy vegetables and herbs.",
    "Crisper Drawer (Right)": "This drawer has low humidity, whisch makes it great for storing produce with lower moisture and longer storage time, like fruits and hardy vegetables.",
    "Door Shelves": "This is the warmest part of the fridge, exposed to temperature changes. It is recommended for items with low sensitivity, low moisture and short storage time.",
    "Upper Freezer Drawers": "It's a slightly warmer part of freezer, so it could be used for storing items that need freezing but with shorter storage durations or for frequent use, like ice cream.",
    "Middle Freezer Drawers": "This drawer is stable and consistently cold. It's recommended for products with medium moisture, medium sensitivity, medium to long storage.",
    "Bottom Freezer Drawers": "It's the coldest area of the freezer. It's great for long-term frozen storage with high moisture, high sensitivity, and long storage time."
}

food_labels = food_fuzzy_values.keys()

# Remove duplicates while preserving order
food_labels = list(OrderedDict.fromkeys(food_labels))

st.cache_data.clear()
st.cache_resource.clear()

# Page setup
st.set_page_config(page_title="Fridge Food Advisor", layout="wide")

# Header
st.title("🧊 Fridge Food Advisor")
st.markdown("Upload a food photo to identify it and get fridge storage advice.")

# Divide layout
col1, col2 = st.columns([1, 2])

zone_images = {
    "Top Shelf": "top_shelf.png",
    "Middle Shelf": "middle_shelf.png",
    "Lower Shelf": "lower_shelf.png",
    "Crisper Drawer (Left)": "crisper_drawer_left.png",
    "Crisper Drawer (Right)": "crisper_drawer_right.png",
    "Door Shelves": "door_shelves.png",
    "Upper Freezer Drawers": "upper_freezer_drawers.png",
    "Middle Freezer Drawers": "middle_freezer_drawers.png",
    "Bottom Freezer Drawers": "bottom_freezer_drawers.png"
}

def get_storage_advice(label):
    for items, vals in food_fuzzy_values.items():
        if label in items:
            shelf_sim.input['moisture'], shelf_sim.input['sensitivity'], shelf_sim.input['storage_time'] = vals
            shelf_sim.compute()
            zone = interpret_shelf(shelf_sim.output['shelf'])
            return zone, f"Store in: {zone}" + ". " + advices[zone]
    return None, "No specific storage advice available."

# In the col1 section (fridge image), show default image first
with col1:
    fridge_img_placeholder = st.empty()
    fridge_img_placeholder.image("fridge.png", caption="Your Fridge", use_container_width=True)

# In col2 where prediction happens
with col2:
    uploaded_file = st.file_uploader("Upload a food image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=False)

        input_tensor = transform(image).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            predicted_idx = output.argmax(1).item()

            if predicted_idx < len(food_labels):
                label = food_labels[predicted_idx]
                st.success(f"**Prediction:** {label.replace('_', ' ').title()}")

                zone, shelf = get_storage_advice(label)
                st.info(shelf)
                # Show specific fridge image if available
                if zone in zone_images:
                    fridge_img_placeholder.image(zone_images[zone], caption=f"Recommended zone: {zone}", use_container_width=True)
            else:
                st.warning("Prediction index out of label range. Check model and label list.")

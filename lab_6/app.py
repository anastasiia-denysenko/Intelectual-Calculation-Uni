import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from collections import OrderedDict
from torchvision import models
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

# Food storage zones
top_shelf = [
    'pad_thai', 'paella', 'pancakes',
    'pho', 'french_onion_soup', 'pizza', 'bibimbap', 'bread_pudding',
    'fried_rice', 'bruschetta', 'ramen', 'risotto', 'chicken_curry', 'hamburger',
    'hot_and_sour_soup', 'hot_dog', 'spaghetti_bolognese', 'spaghetti_carbonara',
    'clam_chowder', 'lasagna', 'club_sandwich', 'tacos', 'macaroni_and_cheese',
    'croque_madame', 'miso_soup', 'nachos'
]

middle_shelf = [
    'eggs_benedict', 'french_toast', 'omelette',
    'grilled_cheese_sandwich', 'breakfast_burrito', 'deviled_eggs'
]

lower_shelf = [
    'beef_carpaccio', 'beef_tartare', 'ceviche', 'filet_mignon', 
    'prime_rib', 'pork_chop', 'steak', 'foie_gras', 'lobster_bisque', 
    'lobster_roll_sandwich'
]

crisper_drawer_left = [
    'beet_salad', 'caesar_salad', 'caprese_salad', 'greek_salad',
    'samosa', 'edamame', 'falafel', 'spring_rolls', 'onion_rings'
]

crisper_drawer_right = [
    'apple_pie', 'strawberry_shortcake', 'carrot_cake', 'cup_cakes',
    'red_velvet_cake', 'macarons', 'baklava', 'chocolate_cake', 
    'chocolate_mousse', 'cheesecake', 'creme_brulee', 'panna_cotta', 
    'beignets', 'cannoli', 'frozen_yogurt', 'tiramisu', 'waffles'
]

door_shelves = [
    'guacamole', 'hummus', 'garlic_bread', 'cheese_plate', 'deviled_eggs',
    'french_fries', 'churros', 'donuts'
]

upper_freezer_drawers = [
    'dumplings', 'gnocchi', 'ravioli', 'takoyaki', 'samosa',
    'gyoza', 'chicken_quesadilla', 'shrimp_and_grits'
]

middle_freezer_drawers = [
    'scallops', 'grilled_salmon', 'mussels', 'oysters',
    'crab_cakes', 'sashimi', 'sushi', 'tuna_tartare'
]

bottom_freezer_drawers = [
    'ice_cream',  'frozen_yogurt'
]

# Combine all for lookup
zones = {
    "Top Shelf": top_shelf,
    "Middle Shelf": middle_shelf,
    "Lower Shelf": lower_shelf,
    "Crisper Drawer (Left)": crisper_drawer_left,
    "Crisper Drawer (Right)": crisper_drawer_right,
    "Door Shelves": door_shelves,
    "Upper Freezer Drawers": upper_freezer_drawers,
    "Middle Freezer Drawers": middle_freezer_drawers,
    "Bottom Freezer Drawers": bottom_freezer_drawers
}


advices = {
    "Top Shelf": " (for ready-to-eat meals, deli items and leftovers)",
    "Middle Shelf": " (for dairy, eggs and drinks)",
    "Lower Shelf": " (for raw meat and fish)",
    "Crisper Drawer (Left)": " (for vegetables)",
    "Crisper Drawer (Right)": " (for fruits and desserts)",
    "Door Shelves": " (for condiments, sauces, juices, butter)",
    "Upper Freezer Drawers": " (for frozen veggies and prepped meals)",
    "Middle Freezer Drawers": " (for frozen meat and seafood)",
    "Bottom Freezer Drawers": " (for ice cream and frozen desserts)"
}

with open("labels.txt", "r") as f:
    food_labels = [line.strip() for line in f]

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

# Modify the function to also return the zone name
def get_storage_advice(label):
    for zone, items in zones.items():
        if label in items:
            return zone, f"Store in: **{zone}**" + advices[zone]
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

                zone, advice_text = get_storage_advice(label)
                st.info(advice_text)

                # Show specific fridge image if available
                if zone in zone_images:
                    fridge_img_placeholder.image(zone_images[zone], caption=f"Recommended zone: {zone}", use_container_width=True)
            else:
                st.warning("Prediction index out of label range. Check model and label list.")
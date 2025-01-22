import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import pandas as pd
import time

# Constants
IMAGE_URL = "https://b3dc-152-58-238-173.ngrok-free.app/capture"  # Replace with your dynamic image URL
MODEL_PATH = "own_data.pt"
REFRESH_INTERVAL = 3  # Time in seconds
PRICES = {
    "kissan_mixed_fruit_jam": 262,
    "amrutanjan": 48,
    "neosprin": 132,
    "fogg_deodrant": 225,
    "custard_powder": 40,
    "dettol_cool_max": 70,
    "vivel_alovera_soap": 60,
    "mtr_hing": 47,
    "mdh_garam_masala": 105,
    "pepsodent_complete_care": 140,
    "mtr_bisebelebath_powder": 155,
    "gopuram_turmeric_powder": 25,
    "ayush_jaggery_powder": 60,
    "mtr_vangibath_powder": 80,
    "maggi_atta_noodles": 14,
    "moms_magic_biscuit": 35,
    "lion_honey": 270,
    "dove_soap": 80,
}

# Initialize YOLO model
model = YOLO(MODEL_PATH)

# Set up Streamlit app
st.title("YOLO Object Detection App")
st.write("Automatically fetch and process images from a changing URL every few seconds.")

# Placeholders for dynamic updates
image_placeholder = st.empty()
table_placeholder = st.empty()

# Function to fetch and process the image
def fetch_and_process_image():
    try:
        # Fetch the image from the URL
        response = requests.get(IMAGE_URL, timeout=5)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))

        # Perform object detection
        results = model.predict(source=image, verbose=False)

        # Display the image
        image_placeholder.image(image, caption="Fetched Image", width=600)

        # Process results and calculate totals
        items = []
        for result in results:
            for box in result.boxes:
                label_index = int(box.cls)
                label_name = result.names[label_index]
                price = PRICES.get(label_name, 0)

                # Add or update item
                existing_item = next((item for item in items if item["ITEM NAME"] == label_name), None)
                if existing_item:
                    existing_item["QUANTITY"] += 1
                    existing_item["AMOUNT"] += price
                else:
                    items.append({"ITEM NAME": label_name, "QUANTITY": 1, "AMOUNT": price})

        # Display the table
        if items:
            total = sum(item["AMOUNT"] for item in items)
            items.append({"ITEM NAME": "Total", "QUANTITY": "", "AMOUNT": total})
            df = pd.DataFrame(items)
            table_placeholder.table(df)
        else:
            table_placeholder.warning("No objects detected.")
    except requests.exceptions.RequestException as e:
        table_placeholder.error(f"Error fetching image: {e}")
    except Exception as e:
        table_placeholder.error(f"An error occurred: {e}")

# Main loop for refreshing content
while True:
    fetch_and_process_image()
    time.sleep(REFRESH_INTERVAL)

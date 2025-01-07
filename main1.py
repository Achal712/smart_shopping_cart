import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os
import time
import pandas as pd

# Constants
IMAGE_URL = "http://172.20.10.4:81/capture"  # Replace with your dynamic image URL
MODEL_PATH = "own_data.pt"
REFRESH_INTERVAL = 3  # Time in seconds
TEMP_IMAGE_PATH = "temp_image.jpg"
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
    "dove_soap": 80
}

# Initialize YOLO model
model = YOLO(MODEL_PATH)

# Set up the Streamlit app
st.title("YOLO Object Detection App")
st.write("Automatically fetch and process images from a changing URL.")

# Loop to keep fetching and processing images
while True:
    try:
        # Fetch the image from the URL
        response = requests.get(IMAGE_URL)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Open the image using PIL
        image = Image.open(BytesIO(response.content))

        # Convert to RGB if the image has an alpha channel
        if image.mode == "RGBA":
            image = image.convert("RGB")

        # Save the image temporarily
        image.save(TEMP_IMAGE_PATH)
        st.image(image, caption="Fetched Image", use_column_width=True)

        # Perform prediction
        with st.spinner("Performing object detection..."):
            results = model.predict(source=TEMP_IMAGE_PATH, verbose=False)

        # Process the results
        items = []
        for result in results:
            for box in result.boxes:
                label_index = int(box.cls)  # Get the class index
                label_name = result.names[label_index]  # Get the class name
                price = PRICES.get(label_name, 0)  # Get the price of the item

                # Update or add item to the list
                existing_item = next((item for item in items if item["ITEM NAME"] == label_name), None)
                if existing_item:
                    existing_item["QUANTITY"] += 1
                    existing_item["AMOUNT"] += price
                else:
                    items.append({"ITEM NAME": label_name, "QUANTITY": 1, "AMOUNT": price})

        # Display the results
        if not items:
            st.warning("No objects detected.")
        else:
            total_amount = sum(item["AMOUNT"] for item in items)
            items.append({"ITEM NAME": "Total", "QUANTITY": "", "AMOUNT": total_amount})
            df = pd.DataFrame(items)
            st.table(df)

        # Clean up the temporary image file
        os.remove(TEMP_IMAGE_PATH)

        # Wait before fetching the next image
        time.sleep(REFRESH_INTERVAL)

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch image from the URL. Error: {e}")
        break
    except Exception as e:
        st.error(f"An error occurred: {e}")
        break

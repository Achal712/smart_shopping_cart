import streamlit as st
from ultralytics import YOLO
from PIL import Image
import requests
from io import BytesIO
import os
import time

# Set up the Streamlit app
st.title("YOLO Object Detection App")
st.write("Automatically fetch and process images from a changing URL.")

# Hardcoded URL for the image
image_url = "http://172.20.10.2:81/capture"  # Replace with your dynamic image URL

# YOLO Model
model = YOLO("own_data.pt")

# Interval to refresh the image (in seconds)
refresh_interval = 5  # Adjust as needed

# Loop to keep fetching and processing the image
while True:
    try:
        # Fetch the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Open the image using PIL
        image = Image.open(BytesIO(response.content))
        st.image(image, caption="Fetched Image", use_column_width=True)

        # Save the image temporarily
        temp_image_path = os.path.join("temp_image.jpg")

        # Convert to RGB if the image has an alpha channel
        if image.mode == "RGBA":
            image = image.convert("RGB")

        image.save(temp_image_path)

        # Perform prediction
        with st.spinner("Performing object detection..."):
            results = model.predict(source=temp_image_path, verbose=False)

        # Initialize a dictionary to count instances of each class
        class_counts = {}

        # Process the results
        for result in results:
            for box in result.boxes:
                label_index = int(box.cls)  # Get the class index
                label_name = result.names[label_index]  # Get the class name
                class_counts[label_name] = class_counts.get(label_name, 0) + 1

        # Display the results
        if not class_counts:
            st.warning("No objects detected.")
        elif len(class_counts) == 1:
            single_class = next(iter(class_counts))  # Get the only key in the dictionary
            st.success(f"Detected one class: {single_class} ({class_counts[single_class]} instance(s)).")
        else:
            total_items = sum(class_counts.values())
            st.success(f"Detected multiple classes: {len(class_counts)} types of items with a total of {total_items} instances.")
            for class_name, count in class_counts.items():
                st.write(f"  - {class_name}: {count} instance(s)")

        # Clean up the temporary image file
        os.remove(temp_image_path)

        # Wait before fetching the next image
        time.sleep(refresh_interval)

    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch image from the URL. Error: {e}")
        break
    except Exception as e:
        st.error(f"An error occurred: {e}")
        break

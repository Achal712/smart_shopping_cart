# Define the list of system-level packages you want to include
packages = [
    "libgl1",  # Required for OpenCV
    "libglib2.0-0",  # For general GLib functionality
    "libsm6",  # For OpenCV GUI operations
    "libxext6",  # X11 miscellaneous extension library
    "libxrender-dev"  # For X11 rendering
]

# Create the packages.txt file and write the packages into it
with open("packages.txt", "w") as file:
    for package in packages:
        file.write(f"{package}\n")

print("packages.txt file created successfully.")

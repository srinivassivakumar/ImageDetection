
import cv2
import numpy as np
import json
import hashlib
import os
import shutil
from skimage.feature import local_binary_pattern

# Base directory where all reference images are located
base_path = r"D:\WCT\ML\magar\crockpattern\testimages"

# Since all images are directly inside base_path, use that as the reference folder
reference_folder = base_path

# Folder where templates will be stored
template_folder = os.path.join(base_path, "Templates")
os.makedirs(template_folder, exist_ok=True)

# Folder where images will be sorted based on similarity
sorted_images_folder = os.path.join(base_path, "Sorted_Images")
os.makedirs(sorted_images_folder, exist_ok=True)

# New image to compare (Modify the path if the image is located elsewhere)
new_image_path = r"D:\WCT\ML\magar\crockpattern\referenceimages\ram.jpg"  # Ensure correct path

# Similarity threshold
SIMILARITY_THRESHOLD = 0.4  # Adjust based on tests


def generate_template(image_path, save_path, use_sift=True):
    """Generate an image template using SIFT/KAZE and Local Binary Pattern (LBP), then save it."""
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        return None

    image = cv2.imread(image_path)
    if image is None:
        print(f"OpenCV Error: Failed to load image at {image_path}")
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Selecting feature detector
    detector = cv2.SIFT_create() if use_sift else cv2.KAZE_create()
    keypoints, descriptors = detector.detectAndCompute(blurred, None)
    descriptors_list = descriptors.tolist() if descriptors is not None else []

    # Local Binary Pattern for texture analysis
    lbp = local_binary_pattern(gray, 24, 3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), range=(0, 256))
    lbp_hist = lbp_hist.tolist()

    hash_value = hashlib.md5(blurred.tobytes()).hexdigest()

    template_data = {
        "keypoints": len(keypoints),
        "descriptors": descriptors_list,
        "lbp": lbp_hist,
        "hash": hash_value
    }

    with open(save_path, "w") as f:
        json.dump(template_data, f)
    print(f"Template saved at: {save_path}")
    return template_data

def compare_with_template(image_path, template_path):
    """Compare a new image with a saved template and return a similarity score."""
    if not os.path.exists(image_path):
        print(f"Error: File not found - {image_path}")
        return 0, False

    image = cv2.imread(image_path)
    if image is None:
        print(f"OpenCV Error: Failed to load image at {image_path}")
        return 0, False

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    detector = cv2.SIFT_create()
    keypoints, descriptors = detector.detectAndCompute(blurred, None)

    if descriptors is None:
        print("No descriptors found in the new image.")
        return 0, False

    if not os.path.exists(template_path):
        print(f"Error: Template file not found - {template_path}")
        return 0, False

    with open(template_path, "r") as f:
        template_data = json.load(f)

    if len(template_data["descriptors"]) == 0:
        print("No descriptors found in the template.")
        return 0, False

    template_descriptors = np.array(template_data["descriptors"], dtype=np.float32)
    descriptors = descriptors.astype(np.float32) if descriptors.dtype != np.float32 else descriptors

    index_params = dict(algorithm=1, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(descriptors, template_descriptors, k=2)
    except cv2.error as e:
        print(f"Error in matching: {e}")
        return 0, False

    good_matches = [m for m, n in matches if m.distance < 0.4 * n.distance]

    similarity = len(good_matches) / max(len(keypoints), template_data["keypoints"])
    print(f"🔍 Similarity Score with {template_path}: {similarity:.2f}")

    new_hash = hashlib.md5(blurred.tobytes()).hexdigest()
    is_same_hash = new_hash == template_data["hash"]
    print(f"Hash Match with {template_path}: {'Yes' if is_same_hash else 'No'}")

    return similarity, is_same_hash

def sort_image(image_path, destination_folder, new_name=None):
    """Move the image to the appropriate folder. Always saves a copy."""
    os.makedirs(destination_folder, exist_ok=True)

    # Rename file if a new name is specified
    if new_name:
        new_file_path = os.path.join(destination_folder, new_name)
    else:
        new_file_path = os.path.join(destination_folder, os.path.basename(image_path))

    shutil.copy(image_path, new_file_path)
    print(f"Image {os.path.basename(image_path)} saved as {new_file_path}")

# Step 1: Generate templates for all images in the reference folder
image_files = [f for f in os.listdir(reference_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(reference_folder, image_file)
    template_path = os.path.join(template_folder, f"{os.path.splitext(image_file)[0]}.json")

    if not os.path.exists(template_path):
        generate_template(image_path, template_path, use_sift=True)

# Step 2: Compare the new image with all templates and sort accordingly
matched_image_paths = []

for image_file in image_files:
    template_path = os.path.join(template_folder, f"{os.path.splitext(image_file)[0]}.json")
    similarity, is_same = compare_with_template(new_image_path, template_path)

    if is_same or similarity > SIMILARITY_THRESHOLD:
        matched_image_paths.append(os.path.join(reference_folder, image_file))

# Step 3: Create a folder named after the new image and save images
new_folder_name = os.path.splitext(os.path.basename(new_image_path))[0]
new_folder_path = os.path.join(sorted_images_folder, new_folder_name)
os.makedirs(new_folder_path, exist_ok=True)

# Save the new image as "reference.jpg" in its own folder
sort_image(new_image_path, new_folder_path, new_name="reference.jpg")

# If matched images are found, add them to the new image’s folder
for matched_image in matched_image_paths:
    sort_image(matched_image, new_folder_path)


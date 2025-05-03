

import cv2
import numpy as np
import json
import os
import shutil
import time
import csv
from skimage.feature import local_binary_pattern
import imagehash
from PIL import Image

# Paths
base_path = r"D:\WCT\ML\magar\crockpattern\wctcroc"
reference_folder = base_path
template_folder = os.path.join(base_path, "Templates")
sorted_images_folder = os.path.join(base_path, "Sorted_Images")
new_image_path = r"D:\WCT\ML\magar\crockpattern\wctcroc\52.jpg"

# Ensure folders exist
os.makedirs(template_folder, exist_ok=True)
os.makedirs(sorted_images_folder, exist_ok=True)

# Similarity threshold
SIMILARITY_THRESHOLD = 0.2

def preprocess_image(image_path):
    print(f"[INFO] Preprocessing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image: {image_path}")
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    return image, enhanced

def generate_template(image_path, save_path):
    if os.path.exists(save_path):
        print(f"[INFO] Template already exists for {image_path}, skipping...")
        return

    print(f"[INFO] Generating template for: {image_path}")
    image, enhanced = preprocess_image(image_path)
    if image is None:
        return

    sift = cv2.SIFT_create()
    kp_sift, des_sift = sift.detectAndCompute(enhanced, None)
    des_sift_list = des_sift.tolist() if des_sift is not None else []

    lbp = local_binary_pattern(enhanced, 24, 3, method='uniform')
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(257), range=(0, 256))
    lbp_hist = lbp_hist.tolist()

    phash_value = str(imagehash.phash(Image.open(image_path)))

    template_data = {
        "keypoints_sift": len(kp_sift),
        "descriptors_sift": des_sift_list,
        "lbp": lbp_hist,
        "phash": phash_value
    }

    with open(save_path, "w") as f:
        json.dump(template_data, f)

    print(f"[SUCCESS] Template saved for {image_path}")

def compare_with_template(image_path, template_path):
    if not os.path.exists(template_path):
        print(f"[WARNING] Template not found for {image_path}, skipping...")
        return 0, False

    print(f"[INFO] Comparing {image_path} with template {template_path}")

    with open(template_path, "r") as f:
        try:
            template_data = json.load(f)
        except json.JSONDecodeError:
            print(f"[ERROR] Corrupt JSON file: {template_path}")
            return 0, False

    if "descriptors_sift" not in template_data or "keypoints_sift" not in template_data:
        print(f"[ERROR] Invalid template format: {template_path}")
        return 0, False

    if not template_data["descriptors_sift"]:
        print(f"[WARNING] No SIFT descriptors found in template: {template_path}")
        return 0, False

    image, enhanced = preprocess_image(image_path)
    if image is None:
        return 0, False

    sift = cv2.SIFT_create()
    kp_sift, des_sift = sift.detectAndCompute(enhanced, None)
    if des_sift is None or len(kp_sift) == 0:
        print(f"[WARNING] No keypoints detected in image: {image_path}")
        return 0, False

    template_des = np.array(template_data["descriptors_sift"], dtype=np.float32)
    des_sift = des_sift.astype(np.float32)

    index_params = dict(algorithm=1, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des_sift, template_des, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.4 * n.distance]
    except cv2.error:
        print(f"[ERROR] FLANN matching failed for {image_path}")
        return 0, False

    similarity = len(good_matches) / max(len(kp_sift), template_data["keypoints_sift"])
    new_phash = str(imagehash.phash(Image.open(image_path)))
    is_same_hash = new_phash == template_data.get("phash", "")

    print(f"[RESULT] Similarity: {similarity:.4f}, pHash Match: {is_same_hash}")
    return similarity, is_same_hash

# Generate Templates
print("[INFO] Generating templates for reference images...")
for img in os.listdir(reference_folder):
    if img.lower().endswith((".jpg", ".png", ".jpeg")):
        template_path = os.path.join(template_folder, f"{os.path.splitext(img)[0]}.json")
        generate_template(os.path.join(reference_folder, img), template_path)

# Compare New Image
print(f"[INFO] Comparing new image: {new_image_path}")
best_match = 0
matched_images = []
match_scores = []
all_comparisons = []
matched_folder = None

for img in os.listdir(reference_folder):
    if not img.lower().endswith((".jpg", ".png", ".jpeg")):
        continue
    template_path = os.path.join(template_folder, f"{os.path.splitext(img)[0]}.json")
    similarity, is_same = compare_with_template(new_image_path, template_path)

    img_path = os.path.join(reference_folder, img)
    all_comparisons.append((img_path, similarity))

    if is_same or similarity > SIMILARITY_THRESHOLD:
        matched_images.append(img_path)
        match_scores.append((img_path, similarity))
        best_match = max(best_match, similarity)

        for folder in os.listdir(sorted_images_folder):
            folder_path = os.path.join(sorted_images_folder, folder)
            if os.path.isdir(folder_path) and img in os.listdir(folder_path):
                matched_folder = folder_path
                break

# Sort New Image
if matched_folder:
    print(f"[INFO] Copying {new_image_path} to existing folder: {matched_folder}")
    shutil.copy(new_image_path, matched_folder)
    csv_folder = matched_folder
else:
    new_folder_name = os.path.splitext(os.path.basename(new_image_path))[0]
    new_folder_path = os.path.join(sorted_images_folder, new_folder_name)
    os.makedirs(new_folder_path, exist_ok=True)
    print(f"[INFO] Creating new folder: {new_folder_path}")
    shutil.copy(new_image_path, os.path.join(new_folder_path, "reference.jpg"))
    for matched_image in matched_images:
        shutil.copy(matched_image, new_folder_path)
    csv_folder = new_folder_path

# Write ALL comparison results to CSV
csv_path = os.path.join(csv_folder, "matches.csv")
with open(csv_path, mode='w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["threshold", SIMILARITY_THRESHOLD])
    writer.writerow([])
    writer.writerow(["reference_image", "image_compared", "match_percentage"])
    for img_path, similarity in all_comparisons:
        writer.writerow([
            os.path.basename(new_image_path),
            os.path.basename(img_path),
            f"{similarity * 100:.2f}%"
        ])

print("[SUCCESS] Image processing completed!")

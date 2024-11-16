# train_cbir.py

import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import pickle

# Paths
IMAGE_FOLDER = 'Label/'  # Folder containing training images
FEATURE_DATABASE = 'feature_database.pkl'    # Path to save feature database
MODEL_PATH = 'segmentation_resnet_model.pth' # Path to save the model

# Load pretrained DeepLab model for segmentation with progress indication
print("Loading DeepLab model for segmentation...")
from torchvision.models.segmentation import DeepLabV3_ResNet101_Weights
segmentation_model = models.segmentation.deeplabv3_resnet101(weights=DeepLabV3_ResNet101_Weights.DEFAULT)
segmentation_model.eval()  # Set model to evaluation mode
print("DeepLab model loaded.")

# Load ResNet model for feature extraction with progress indication
print("Loading ResNet model for feature extraction...")
from torchvision.models import ResNet50_Weights
feature_extractor = models.resnet50(weights=ResNet50_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(feature_extractor.children())[:-1])  # Remove final classification layer
feature_extractor.eval()
print("ResNet model loaded.")

# Define image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Function to segment an image and return segmentation map
def segment_image(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = segmentation_model(input_tensor)["out"][0]  # Get segmentation output
    segmentation = output.argmax(0).numpy()  # Convert to numpy array
    return image, segmentation

# Function to extract regions from an image based on segmentation mask
def extract_regions(image, segmentation):
    regions = {}
    unique_labels = np.unique(segmentation)
    for label in unique_labels:
        mask = segmentation == label
        masked_image = Image.fromarray((np.array(image) * mask[:, :, None]).astype(np.uint8))
        regions[label] = masked_image  # Save each region by label
    return regions

# Function to extract features from a region image using ResNet
def extract_features(image):
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(input_tensor)
    return features.squeeze().numpy()  # Convert to numpy array for CBIR

# Dictionary to store features for each image and region
image_features_db = {}

# Process all images in folder and extract region-based features
print("Starting feature extraction for all images in the folder...")

for i, image_name in enumerate(os.listdir(IMAGE_FOLDER)):
    image_path = os.path.join(IMAGE_FOLDER, image_name)
    print(f"Processing image {i+1}/{len(os.listdir(IMAGE_FOLDER))}: {image_name}")
    
    # Segment the image
    image, segmentation = segment_image(image_path)
    print(f" - Segmentation completed for {image_name}")
    
    # Extract regions and their features
    regions = extract_regions(image, segmentation)
    image_features_db[image_name] = {}
    
    for label, region_image in regions.items():
        features = extract_features(region_image)
        image_features_db[image_name][label] = features
        print(f"   - Features extracted for region {label} in {image_name}")

# Save the feature database for future testing
with open(FEATURE_DATABASE, 'wb') as f:
    pickle.dump(image_features_db, f)
print("Feature database saved.")

# Save both models for future testing
torch.save({
    'segmentation_model': segmentation_model.state_dict(),
    'feature_extractor': feature_extractor.state_dict(),
}, MODEL_PATH)
print("Models saved. Training complete.")

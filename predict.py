# predict.py
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sklearn.preprocessing import StandardScaler
import joblib
import argparse

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SVM_PATH = 'svm_sensitive_content_model.pkl'
SCALER_PATH = 'scaler.pkl'
RESNET_PATH = 'resnet_feature_extractor.pth'

# --- Prediction Logic ---
class Predictor:
    def __init__(self, device, svm_path, scaler_path, resnet_path):
        self.device = device
        
        # Load classification model
        self.clf = joblib.load(svm_path)
        self.scaler = joblib.load(scaler_path)

        # Load feature extractor
        self.resnet = models.resnet18(weights=None)
        self.resnet.fc = nn.Identity()
        self.resnet.load_state_dict(torch.load(resnet_path, map_location=device))
        self.resnet = self.resnet.to(device)
        self.resnet.eval()
        
        # Image transforms
        self.data_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path):
        """Classifies a single image."""
        try:
            image = Image.open(image_path).convert('RGB')
        except FileNotFoundError:
            return f"Error: Image not found at {image_path}"

        img_tensor = self.data_transforms(image).unsqueeze(0).to(self.device)

        # 1. Extract features
        with torch.no_grad():
            features = self.resnet(img_tensor).cpu().numpy()

        # 2. Scale features and predict
        features_scaled = self.scaler.transform(features)
        prediction = self.clf.predict(features_scaled)

        # SVM returns 1 for inliers (sensitive) and -1 for outliers (not sensitive)
        return "SENSITIVE" if prediction[0] == 1 else "NOT SENSITIVE"

# --- Command-Line Interface ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Classify an image as SENSITIVE or NOT SENSITIVE.")
    parser.add_argument('image', type=str, help="Path to the image file.")
    args = parser.parse_args()
    
    predictor = Predictor(DEVICE, SVM_PATH, SCALER_PATH, RESNET_PATH)
    result = predictor.predict_image(args.image)
    
    print(f"Image: {args.image}")
    print(f"Prediction: {result}")
# train.py
import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import joblib

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = 'dataset/train' # Directory containing the 'sensitive' folder
BATCH_SIZE = 32

# --- Data Transformations ---
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Main Training Logic ---
def train():
    print(f"Using device: {DEVICE}")

    # 1. Load Dataset
    print("Loading dataset...")
    dataset = datasets.ImageFolder(DATA_DIR, transform=data_transforms)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load ResNet Feature Extractor
    print("Loading ResNet-18 feature extractor...")
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    resnet.fc = nn.Identity()  # Remove classifier head to get feature vectors
    resnet = resnet.to(DEVICE)
    resnet.eval()

    # 3. Extract Features
    features_list = []
    print("Extracting features from sensitive content images...")
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(DEVICE)
            outputs = resnet(inputs)
            features_list.append(outputs.cpu().numpy())
    
    features = np.vstack(features_list)

    # 4. Normalize Features
    print("Normalizing features...")
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # 5. Train One-Class SVM
    print("Training One-Class SVM on sensitive content features...")
    # The `nu` parameter is an upper bound on the fraction of training errors 
    # and a lower bound of the fraction of support vectors.
    clf = svm.OneClassSVM(kernel='rbf', gamma='auto', nu=0.1).fit(features_scaled)

    # 6. Save Models and Scaler
    print("Saving models...")
    joblib.dump(clf, 'svm_sensitive_content_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    torch.save(resnet.state_dict(), 'resnet_feature_extractor.pth')

    print("\nTraining complete. Model components saved successfully:")
    print("- resnet_feature_extractor.pth")
    print("- svm_sensitive_content_model.pkl")
    print("- scaler.pkl")

if __name__ == '__main__':
    train()
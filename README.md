# CensorFlow – Visual Threat Detector
Aegis Vision is an AI-powered content moderation tool designed to identify sensitive visual content. It uses a sophisticated anomaly detection approach to classify images as "SENSITIVE" or "NOT SENSITIVE", making it a robust solution for platforms needing to regulate user-generated content.

## Core Concept
Instead of a traditional binary classifier that requires large datasets for both "sensitive" and "not sensitive" categories, this project employs a One-Class Support Vector Machine (SVM).

The model is trained exclusively on the features of sensitive content. It learns the characteristics of this class and creates a boundary around it. Any new image that falls outside this learned boundary is considered an anomaly, and therefore, "NOT SENSITIVE". This approach is highly effective because the "not sensitive" category is infinitely vast and diverse, making it difficult to model directly.

## How It Works
The system is composed of two main stages:

Feature Extraction: A pre-trained ResNet-18 model, with its final classification layer removed, is used to analyze an input image and convert it into a high-dimensional feature vector (a numerical representation).

Classification: The One-Class SVM takes this feature vector and determines if it belongs to the distribution of sensitive content it learned during training.

## Directory Structure
Organize your project files and folders as shown below for the scripts to work correctly.

    .
    ├── dataset/
    │   └── train/
    │       └── sensitive/
    │           ├── image1.jpg
    │           └── image2.png
    │
    ├── train.py                    # Script to train the models
    ├── predict.py                  # Script to classify a new image
    ├── requirements.txt            # Project dependencies
    └── README.md

## Installation
1.  Clone the repository:
```
git clone <your-repo-link>
cd <your-repo-name>
```
2.  Create a virtual environment (recommended):
```
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3.  Install the required dependencies:
```
pip install -r requirements.txt
```


## Usage
The process is divided into two parts: training the model on your data and then using the trained model to make predictions.

### 1. Training the Model
First, you need to train the feature extractor and the SVM.

#### Step 1: Prepare your data
Gather all your sensitive training images and place them inside the ```dataset/train/sensitive/``` directory.

#### Step 2: Run the training script
Open your terminal and run:
```
python train.py
```
This will create three files in your root directory: ```resnet_feature_extractor.pth```, ```svm_sensitive_content_model.pkl```, and ```scaler.pkl```.


### 2. Classifying an Image
Once the models are trained, you can use them to classify any new image.

#### Run prediction

From your terminal, run the script followed by the path to the image you want to classify.
```
python predict.py /path/to/your/image.jpg
```

#### Example Output:
```
Image: /path/to/your/image.jpg
Prediction: SENSITIVE
```

### Model Components
* ```resnet_feature_extractor.pth```: The saved state of the pre-trained ResNet-18 model used for feature extraction.

* ```svm_sensitive_content_model.pkl```: The trained One-Class SVM classifier object.

* ```scaler.pkl```: The StandardScaler object fitted on the training features, necessary to normalize new data before prediction.

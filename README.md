# Retinal-Disease-Detection-and-Classification
1. Project Overview
This project develops a deep learning-based system for the automated detection and classification of common retinal diseases from fundus images. Leveraging Convolutional Neural Networks (CNNs) with TensorFlow, the system can classify images into four categories: Normal, Diabetic Retinopathy, Cataract, and Glaucoma. This automated diagnostic aid can assist ophthalmologists and healthcare professionals in early screening and detection, particularly in resource-limited settings.

2. Problem Statement
Early and accurate detection of retinal diseases like Diabetic Retinopathy, Cataract, and Glaucoma is crucial for preventing vision loss. However, manual diagnosis by ophthalmologists can be time-consuming and requires specialized expertise, which may not be readily available everywhere. An automated system can provide a quick, preliminary screening tool, improving accessibility and efficiency in eye care.

3. Objective
Understand and Preprocess Data: Load, resize, and normalize retinal image data.

Build a Deep Learning Model: Design and implement a Convolutional Neural Network (CNN) using TensorFlow/Keras for multi-class image classification.

Train and Evaluate Model: Train the CNN on the prepared dataset and thoroughly evaluate its performance using metrics like accuracy, confusion matrix, and classification report.

Demonstrate Prediction: Show real-time classification on unseen retinal images.

4. Dataset
The dataset used in this project consists of approximately 4000 retinal images categorized into four distinct classes:

normal: Approximately 1074 images of healthy retinas.

cataract: Approximately 1038 images showing signs of cataracts.

glaucoma: Approximately 1007 images indicative of glaucoma.

diabetic_retinopathy: Approximately 1098 images showing signs of diabetic retinopathy.

These images were collected from various sources, including IDRiD, Oculur recognition, and HRF datasets.

Dataset Structure:
The dataset is expected to be organized in a directory named dataset/ with subfolders for each class:

dataset/
├── normal/
│   ├── image_001.jpg
│   └── ...
├── cataract/
│   ├── image_a.jpg
│   └── ...
├── glaucoma/
│   ├── image_x.jpg
│   └── ...
└── diabetic_retinopathy/
    ├── image_p.jpg
    └── ...
5. Methodology
The project follows a standard deep learning workflow for image classification:

Data Loading & Preprocessing
Images are loaded from their respective class subfolders using opencv-python (cv2).

All images are resized to a uniform (128, 128) pixels to serve as consistent input for the CNN.

Pixel values are normalized to a [0, 1] range for optimal model training.

Labels (class names) are encoded into numerical format using LabelEncoder.

The dataset is split into training (80%) and testing (20%) sets, ensuring stratification to maintain class distribution.

Data Augmentation
ImageDataGenerator from Keras is used to apply various random transformations (rotation, shifts, shear, zoom, horizontal flips) to the training images. This technique artificially expands the training dataset, improving model generalization and reducing overfitting.

CNN Model Architecture
A Sequential Keras model is built, featuring:

Multiple Conv2D layers with ReLU activation for feature extraction.

MaxPooling2D layers for dimensionality reduction and translation invariance.

A Flatten layer to convert 2D feature maps into a 1D vector.

Dense (fully connected) layers with ReLU activation.

Dropout layers for regularization to prevent overfitting.

A final Dense layer with softmax activation for multi-class probability output.

Model Training
The model is compiled using the Adam optimizer (with a learning rate of 0.001) and sparse_categorical_crossentropy loss function (suitable for integer-encoded labels).

The model is trained for 15 epochs using the augmented training data from ImageDataGenerator and validated on the untouched test set.

Model Evaluation
The trained model's performance is evaluated on the test set using:

Test Loss and Accuracy: Overall performance metrics.

Confusion Matrix: A visual representation of correct and incorrect classifications for each class.

Classification Report: Provides detailed metrics including Precision, Recall, and F1-Score for each class, along with overall accuracy.

Training history (accuracy and loss over epochs) is plotted to visualize model learning and identify potential overfitting.

Demonstrate Prediction
A random sample image from the test set is selected.

The model predicts the class of this sample image, and the predicted label and its probability are displayed alongside the true label, offering a real-world prediction example.

6. Key Findings & Model Performance

The model demonstrated strong capabilities in distinguishing between the four retinal conditions.

Overall Accuracy: Achieved a validation accuracy of approximately 88-92%.

Confusion Matrix Insights: Revealed high true positive rates for Normal and Diabetic Retinopathy, but some confusion between Cataract and Glaucoma.

Classification Report: Showed balanced precision and recall across most classes, indicating robust performance.

This project highlights the potential of deep learning in automating medical image diagnostics, offering a scalable solution for preliminary screening.

7. How to Run the Project
Prerequisites:

Python 3.x installed.

Google Colab or a Python IDE.

Stable internet connection (for downloading dependencies and potentially dataset).

Download and Prepare Dataset:

Crucial: Obtain the retinal image dataset with normal, cataract, glaucoma, and diabetic_retinopathy subfolders.

Zip the dataset folder: On your local machine, right-click the dataset folder and compress it into a .zip file (e.g., dataset.zip).

Upload to Google Colab:

Open your Colab notebook.

In the left sidebar, click the "Files" icon.

Click the "Upload to session storage" icon (file with arrow up).

Select your dataset.zip file and upload it.

Once uploaded, run the following cell in Colab to unzip:

!unzip /content/dataset.zip -d /content/

Verify extraction by running !ls /content/dataset/.

Open and Run in Google Colab:

Open a new Google Colab notebook.

Copy the entire Python code from this README's immersive code block into a code cell in your Colab notebook.

Run all cells sequentially from top to bottom. The first cell will install necessary libraries.

Local Setup (Alternative to Colab):

Clone/Download: Get the project files (Python script, requirements.txt).

Virtual Environment:

python -m venv venv
# On Windows: .\venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

Install Dependencies:

pip install -r requirements.txt

Place Dataset: Ensure your dataset folder (with subfolders) is in the same directory as your Python script.

Run Script: python retinal_disease_detection.py

8. Files in the Repository
dataset/: Directory containing the retinal image data (subfolders: normal, cataract, glaucoma, diabetic_retinopathy).

retinal_disease_detection.py: The main Python script containing all the code for the project.

README.md: This file, providing project documentation.

requirements.txt: Lists all Python library dependencies.

9. Future Work
Larger/More Diverse Dataset: Train on a significantly larger and more varied dataset for improved generalization.

Advanced CNN Architectures: Experiment with pre-trained models (Transfer Learning) like ResNet, VGG, Inception, or EfficientNet for potentially higher accuracy and faster convergence.

Imbalanced Data Handling: Implement techniques like class weighting, oversampling (SMOTE), or undersampling if class distribution is skewed.

Explainable AI (XAI): Integrate techniques like Grad-CAM or LIME to visualize which parts of the image the model focuses on for its predictions, enhancing trust and interpretability.

Web Application/Mobile App Deployment: Build a user-friendly web interface (e.g., with Flask, Streamlit) or a mobile application to allow real-time image uploads and predictions.

Integration with Medical Systems: Explore potential for integration with Electronic Health Record (EHR) systems (requires compliance and security considerations).

# Tuberculosis-Detection-Using-Deep-Learning

**Project Overview**
Tuberculosis Detection Using Deep Learning is a medical imaging project aimed at identifying tuberculosis (TB) from chest X-ray images using deep learning models. TB is a highly contagious disease, and early diagnosis can significantly help in controlling its spread. This project leverages transfer learning techniques with pre-trained CNN models like VGG16, ResNet50, and EfficientNetB0 to classify X-rays into TB Positive or Normal classes with high accuracy.

**Objectives**
* Develop a deep learning-based classification system to detect TB in chest X-rays.
* Compare and evaluate the performance of multiple CNN architectures.
* Implement a Streamlit web app to make predictions user-friendly and accessible.
* Deploy the model using AWS EC2 for public accessibility.

  **Dataset**
The dataset used is a curated collection of Tuberculosis Chest X-ray Images, containing:

- Normal Chest X-rays
- TB Chest X-rays

**Deep Learning Models**

This project evaluates and compares three pre-trained models with transfer learning:

Model	Validation Accuracy
✅ VGG16	97.91%
ResNet50	86.22%
EfficientNetB0	82.90%

 **Techniques Used**
 - Transfer Learning
 - Dropout Regularization
 - Data Augmentation (rotation, zoom, flip, etc.)
 - Binary Crossentropy Loss
 - Adam Optimizer

**Evaluation Metrics**
Accuracy
Precision
Recall
F1-Score
ROC-AUC
Confusion Matrix
Precision-Recall Curve

**Web Application**
A Streamlit-based user interface has been developed where users can upload a chest X-ray image and get a prediction result (TB or Normal).

🔗 Web App Features:
Upload X-ray image
Display prediction with probability
Model interpretability using Grad-CAM (optional enhancement)

-Deployed on AWS EC2 (Ubuntu) instance using Nginx and Gunicorn for scalability.

**Project Structure**

📦Tuberculosis_Detection/
 ┣  📂basedir/
 ┃ ┣ test/
 ┃ ┣ train/
 ┃ ┗ validation/
 ┣ 📂Dataset/
 ┃ ┣ 📂Normal Chest X-rays/
 ┃ ┗ 📂TB Chest X-rays/
 ┣ 📂sample_images/
 ┃ ┣ Normal_1
 ┃ ┣ TB_1
 ┣ 📂saved_models/
 ┃ ┣ vgg16_model.keras
 ┃ ┣ resnet50_model.keras
 ┃ ┗ efficientnetb0_model.keras
 ┣ app.py
 ┣ TB_detect.ipynb
 ┣ 📄README.md
 ┣ 📄requirements.txt
 
**How to Run Locally**

1. Clone the Repository 
git clone https://github.com/yourusername/tuberculosis-detection.git
cd tuberculosis-detection

2. Install Requirements
   pip install -r requirements.txt
3. Launch Streamlit App
    streamlit run app/app.py

**Deployment (AWS EC2)**
Launch an Ubuntu EC2 instance
Upload the project files
Setup Python environment and dependencies
Use gunicorn and nginx for production deployment
Make sure port 80 is open in the security group

**Results**
VGG16 emerged as the best-performing model with nearly 98% accuracy.
Demonstrated effective binary classification for medical diagnosis.
Successfully deployed the prediction system on a cloud platform.

**Conclusion**
This project demonstrates the power of deep learning in assisting radiologists with TB detection. The developed system shows high accuracy, especially with VGG16, and is deployed in a user-friendly manner through a web interface.

**Acknowledgements**
Dataset source: [Kaggle / Open Source TB X-rays]
Streamlit community for web development resources

**Contact**
 Uma Rajesh 
For queries 
📧 umarajesh2809@gmail.com
🔗 LinkedIn Profile www.linkedin.com/in/uma-rajesh [GitHub Profile]

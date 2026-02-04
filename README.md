# 🍅 Tomato Plant Disease Detection using Deep Learning

This project detects diseases in tomato plant leaves using a Convolutional Neural Network (CNN). It uses a Kaggle dataset and provides a user-friendly web interface built with Streamlit. The application can also be deployed using Docker for easy sharing and demonstration.

---

## 📌 Features

- Tomato leaf disease classification (11 classes)
- CNN model trained using TensorFlow / Keras
- Upload leaf images and get real-time predictions
- Streamlit-based interactive GUI
- Dockerized for consistent deployment
- Clean GitHub repository structure with screenshots and documentation

---

## 📂 Project Structure

tomato-disease-detection/
|

├── app.py    # Streamlit application

├── Dockerfile             # Docker deployment

├── requirements.txt       # Python dependencies

├── README.md              # Project documentation

├── tomato_model.h5        # Trained CNN model

├── src/                   # Training scripts and preprocessing

│   ├── train.py

│   ├── model.py

│   └── preprocess.py

├── screenshots/           # Screenshots for README & PPT



---

## 📊 Dataset

- **Source:** Kaggle – Tomato Plant Disease Dataset
- **Total Classes:** 11 (healthy + diseased)
- **Dataset is not included due to size constraints**

**Classes:**
Bacterial_spot, Early_blight, Late_blight, Leaf_Mold, Septoria_leaf_spot,
Spider_mites Two-spotted_spider_mite, Target_Spot, Tomato_Yellow_Leaf_Curl_Virus,
Tomato_mosaic_virus, healthy, powdery_mildew

---

## 🧠 Model Details

- **Model Type:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow / Keras
- **Image Size:** 224 × 224
- **Optimizer:** Adam
- **Loss Function:** Categorical Crossentropy
- **Epochs:** 10
- **Batch Size:** 32
- **Validation Accuracy:** ~78–82%

---

## 📈 Epoch-wise Training Summary

| Epoch | Training Accuracy | Validation Accuracy | Training Loss | Validation Loss |
|-------|-------------------|--------------------|---------------|-----------------|
| 1     | 42%               | 49%                | 1.61          | 1.52            |
| 2     | 60%               | 66%                | 1.10          | 0.94            |
| 3     | 65%               | 74%                | 0.97          | 0.75            |
| 4     | 69%               | 71%                | 0.87          | 0.93            |
| 5     | 71%               | 76%                | 0.81          | 0.67            |
| 6     | 73%               | 75%                | 0.76          | 0.84            |
| 7     | 73%               | 77%                | 0.74          | 0.71            |
| 8     | 75%               | 78%                | 0.68          | -               |
| 9     | 76%               | 79%                | -             | -               |
| 10    | 77–80%            | 78–80%             | -             | -               |

**Observations:**
- Steady increase in accuracy shows effective learning.
- Validation accuracy closely tracks training accuracy → good generalization.
- Minor fluctuations in validation loss are expected due to dataset diversity.
- CPU-only training caused longer epoch durations (~30–40 min per epoch).

**Limitations:**
- Accuracy can improve with transfer learning (MobileNetV2, ResNet50) and GPU acceleration.
- Model is trained on a fixed dataset and may need retraining for real-world scenarios.

---

## 🖼️ Screenshots

- Model Training
- Streamlit Interface
- Prediction Result

---

## 🚀 How to Run

### Setup (Without Docker)

```bash
git clone https://github.com/DudalaShrujana/tomato-disease-detection.git
cd tomato-disease-detection
pip install -r requirements.txt
streamlit run app.py
```

Open browser: `http://localhost:8501`

### Docker Deployment

```bash
# Build Docker Image
docker build -t tomato-disease-app .

# Run Container
docker run -d -p 8501:8501 tomato-disease-app
```

Open browser: `http://localhost:8501`

---

## 📦 Future Scope

- Improve accuracy using transfer learning (MobileNetV2, ResNet50)
- GPU acceleration for faster training
- Web interface improvements (image gallery, batch prediction)
- Extend to other crops and plant diseases

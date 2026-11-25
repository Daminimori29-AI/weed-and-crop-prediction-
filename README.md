# 🌱 AI-Based Crop and Weed Species Detection

## About Project
This project uses Deep Learning and Convolutional Neural Networks (CNN) to identify crop and weed species from plant images.

Users can upload a plant image and get the predicted species, crop/weed classification, confidence score, and relevant precautions.

## Features
- Identifies 12 plant species.
- Classifies plants as crops or weeds.
- Displays prediction confidence.
- Provides crop precautions and weed control methods.
- Includes herbicide safety instructions.
- Simple and user-friendly Streamlit interface.

## Dataset
**Plant Seedlings Dataset V2**

Source: [Kaggle Dataset](https://www.kaggle.com/datasets/vbookshelf/v2-plant-seedlings-dataset)

- 3 Crop Classes
- 9 Weed Classes
- Training and Validation Split: 80:20

## Technologies Used
- Python
- TensorFlow / Keras
- CNN
- Streamlit
- NumPy
- Pillow (PIL)

## Model
A CNN-based deep learning model trained for 25 epochs to classify 12 plant species.

## How to Run

Install dependencies:

```bash
pip install streamlit tensorflow numpy pillow
```

Run the application:

```bash
streamlit run app.py
```

## Future Scope
- Improve model accuracy.
- Add more plant species.
- Provide detailed species-specific precautions.
- Develop a mobile application.

## Disclaimer
This project is developed for educational purposes. Predictions should be verified before making agricultural decisions.

---

**Developed as an academic project**  
Department of Artificial Intelligence and Data Science  
Chameli Devi Group of Institutions, Indore

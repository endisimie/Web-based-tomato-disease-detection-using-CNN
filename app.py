import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from PIL import Image

# Load your trained model
MODEL_PATH = "tomatoes97.h5"  # Replace with your model file
model = load_model(MODEL_PATH)

# Define class names (these should match the training dataset class names)
class_names = ["Healthy", "Early Blight", "Late Blight"]  # Update with your class names

class_names = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

# Define suggestions for each class
class_recommendations = {
    "Tomato___Bacterial_spot": """
    **Expert Recommendation:**
    - **Cause**: Caused by the bacterium *Xanthomonas campestris*.
    - **Symptoms**: Water-soaked spots on leaves and fruits, yellow halos.
    - **Prevention**: Use disease-free seeds, practice crop rotation, and avoid overhead irrigation.
    - **Treatment**: Apply copper-based fungicides. Remove and destroy infected plant debris.
    """,
    "Tomato___Early_blight": """
    **Expert Recommendation:**
    - **Cause**: Caused by the fungus *Alternaria solani*.
    - **Symptoms**: Dark, concentric spots on older leaves; yellowing and leaf drop.
    - **Prevention**: Use resistant varieties and practice crop rotation. Avoid wetting the foliage.
    - **Treatment**: Use fungicides containing chlorothalonil or mancozeb. Remove infected leaves.
    """,
    "Tomato___Late_blight": """
    **Expert Recommendation:**
    - **Cause**: Caused by the oomycete *Phytophthora infestans*.
    - **Symptoms**: Water-soaked lesions on leaves, stems, and fruits, followed by rapid collapse.
    - **Prevention**: Use resistant varieties, improve air circulation, and avoid overhead watering.
    - **Treatment**: Use systemic fungicides like metalaxyl or fosetyl-aluminum. Destroy infected plants.
    """,
    "Tomato___Leaf_Mold": """
    **Expert Recommendation:**
    - **Cause**: Caused by the fungus *Cladosporium fulvum*.
    - **Symptoms**: Yellow spots on upper leaf surfaces, olive-green mold on undersides.
    - **Prevention**: Ensure good ventilation and avoid overwatering.
    - **Treatment**: Use fungicides like mancozeb or copper hydroxide. Remove affected leaves.
    """,
    "Tomato___Septoria_leaf_spot": """
    **Expert Recommendation:**
    - **Cause**: Caused by the fungus *Septoria lycopersici*.
    - **Symptoms**: Small, circular spots with dark borders and gray centers.
    - **Prevention**: Use resistant varieties and rotate crops. Avoid wetting leaves during watering.
    - **Treatment**: Apply fungicides like chlorothalonil or copper-based products. Remove infected foliage.
    """,
    "Tomato___Spider_mites Two-spotted_spider_mite": """
    **Expert Recommendation:**
    - **Cause**: Infestation by *Tetranychus urticae* mites.
    - **Symptoms**: Yellow stippling on leaves, webbing, and leaf drop.
    - **Prevention**: Increase humidity and use natural predators like ladybugs.
    - **Treatment**: Apply insecticidal soaps, neem oil, or specific miticides.
    """,
    "Tomato___Target_Spot": """
    **Expert Recommendation:**
    - **Cause**: Caused by the fungus *Corynespora cassiicola*.
    - **Symptoms**: Brown lesions with concentric rings, defoliation, and fruit spotting.
    - **Prevention**: Use resistant varieties and avoid wetting leaves during watering.
    - **Treatment**: Apply fungicides like azoxystrobin or chlorothalonil. Remove infected plant debris.
    """,
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": """
    **Expert Recommendation:**
    - **Cause**: Transmitted by whiteflies carrying the *Begomovirus*.
    - **Symptoms**: Yellowing, curling leaves, stunted growth, and reduced fruit yield.
    - **Prevention**: Control whiteflies with sticky traps and insecticides. Use resistant varieties.
    - **Treatment**: Remove infected plants to prevent further spread.
    """,
    "Tomato___Tomato_mosaic_virus": """
    **Expert Recommendation:**
    - **Cause**: Spread mechanically through tools and contact with infected plants.
    - **Symptoms**: Mottled, mosaic-like patterns on leaves; distorted fruit growth.
    - **Prevention**: Sterilize tools and practice good hygiene. Use virus-free seeds.
    - **Treatment**: Remove and destroy infected plants. There is no chemical cure.
    """,
    "Tomato___healthy": """
    **Expert Recommendation:**
    - **Status**: The plant is healthy. Maintain optimal care.
    - **Tips**: Continue to provide adequate sunlight, water, and nutrients. Monitor regularly for signs of pests or diseases.
    """
}
# Set page configuration
st.set_page_config(page_title="Tomato Disease Detection", layout="wide")

# Header
st.markdown(
    """
    <div style="padding:10px;">
    <h1 style="text-align:center;">Tomato Disease Detection - AI Powered</h1>
    <p style="text-align:center;">A tool for identifying diseases in tomato plants and providing expert recommendations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for project results and suggestions
st.sidebar.header("Project Insights & Suggestions")
st.sidebar.markdown(
    """
    ### Insights
    - This application detects diseases in tomato leaves using a DenseNet-trained model with 94 % of accuracy.
    - Upload an image, and the app will predict the health status with 9 different disease of the leaf.
    
    ### Suggestions
    - Ensure the uploaded image is clear and well-lit.
    - If diseased, take appropriate measures like removing affected leaves or applying fungicides.
    """
)
# File uploader
uploaded_file = st.file_uploader("Upload a Tomato Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    original_width, original_height = image.size
    
    # Resize if the image is too large
    if original_width > 600 or original_height > 600:
        max_size = 600
        image.thumbnail((max_size, max_size))  # Maintain aspect ratio
    
    # Display resized image
    st.image(image, caption="Uploaded Image")

    # Preprocess the image
    def preprocess_image(img):
        img = img.resize((256, 256))  # Resize to match model input
        img_array = img_to_array(img)  # Convert to array
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array

    # Make predictions
    st.write("Processing the image...")
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)

    # Determine the predicted class
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = class_names[predicted_class_index]

    # Display the results
    st.success(f"**Predicted Class:** {predicted_class_name}")

    # Display class-specific suggestion
    st.markdown("### Suggestion Based on Prediction:")
    st.info(class_recommendations.get(predicted_class_name, "No specific suggestions available."))
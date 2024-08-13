import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import cv2

# Load models
@st.cache(allow_output_mutation=True)
def load_models():
    cnn_model = tf.keras.models.load_model('models/cnn_model.h5')
    transfer_model = tf.keras.models.load_model('models/transfer_model.h5')
    with open('models/rf_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    with open('models/svm_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open('models/gb_model.pkl', 'rb') as f:
        gb_model = pickle.load(f)
    return cnn_model, transfer_model, rf_model, svm_model, gb_model

cnn_model, transfer_model, rf_model, svm_model, gb_model = load_models()

def preprocess_image(image):
    img = image.resize((150, 150))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def predict(image, model):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    return 'Pneumonia' if prediction[0][0] > 0.5 else 'Normal'

def predict_traditional(image, model):
    img_array = preprocess_image(image)
    flattened = img_array.reshape(1, -1)
    prediction = model.predict(flattened)
    return 'Pneumonia' if prediction[0] > 0.5 else 'Normal'

def predict_with_confidence(image, model):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)[0][0]
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return 'Pneumonia' if prediction > 0.5 else 'Normal', confidence

# Update the main function to use predict_with_confidence and display confidence scores

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(image, heatmap, alpha=0.4):
    img = img_to_array(image)
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

def find_similar_cases(image, num_cases=5):
    # This function would need a database of labeled images to compare against
    # For simplicity, let's assume we have a function that returns similar cases
    similar_cases = get_similar_cases(image, num_cases)
    return similar_cases

def ensemble_predict(image):
    predictions = [
        predict(image, cnn_model),
        predict(image, transfer_model),
        predict_traditional(image, rf_model),
        predict_traditional(image, svm_model),
        predict_traditional(image, gb_model)
    ]
    return max(set(predictions), key=predictions.count)



def main():
    st.title('Pneumonia Detection from Chest X-Rays')

    st.write("""
    This app uses various machine learning models to detect pneumonia from chest X-ray images.
    Upload an image and see the predictions from different models!
    """)

    uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded X-ray image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("CNN Model")
            cnn_prediction = predict(image, cnn_model)
            st.write(f"Prediction: {cnn_prediction}")

            st.subheader("Transfer Learning Model")
            transfer_prediction = predict(image, transfer_model)
            st.write(f"Prediction: {transfer_prediction}")

        with col2:
            st.subheader("Random Forest")
            rf_prediction = predict_traditional(image, rf_model)
            st.write(f"Prediction: {rf_prediction}")

            st.subheader("SVM")
            svm_prediction = predict_traditional(image, svm_model)
            st.write(f"Prediction: {svm_prediction}")

            st.subheader("Gradient Boosting")
            gb_prediction = predict_traditional(image, gb_model)
            st.write(f"Prediction: {gb_prediction}")

        # Visualization
        st.subheader("Model Comparison")
        models = ['CNN', 'Transfer Learning', 'Random Forest', 'SVM', 'Gradient Boosting']
        predictions = [cnn_prediction, transfer_prediction, rf_prediction, svm_prediction, gb_prediction]
        colors = ['green' if pred == 'Normal' else 'red' for pred in predictions]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=models, y=[1]*5, palette=colors)
        plt.yticks([])
        plt.xticks(rotation=45)
        plt.title('Model Predictions')
        for i, pred in enumerate(predictions):
            plt.text(i, 0.5, pred, ha='center', va='center')
        st.pyplot(fig)
    
    if st.checkbox('Show Grad-CAM for CNN'):
    img_array = preprocess_image(image)
    heatmap = make_gradcam_heatmap(img_array, cnn_model, 'conv2d_2')
    gradcam_image = display_gradcam(image, heatmap)
    st.image(gradcam_image, caption='Grad-CAM Visualization', use_column_width=True)

    if st.checkbox('Show Similar Cases'):
    similar_cases = find_similar_cases(image)
    for i, case in enumerate(similar_cases):
        st.image(case['image'], caption=f"Similar Case {i+1}: {case['label']}", width=200)

    st.subheader("Ensemble Prediction")
    ensemble_prediction = ensemble_predict(image)
    st.write(f"Ensemble Prediction: {ensemble_prediction}")

    
if __name__ == '__main__':
    main()
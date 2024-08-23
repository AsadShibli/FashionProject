from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import faiss

app = Flask(__name__)

# Load the pretrained model (EfficientNetB1)
model = tf.keras.applications.EfficientNetB1(weights='imagenet', include_top=False, pooling='avg')

def preprocess_image(image):
    img = tf.image.resize(image, [224, 224])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img

# Load saved embeddings and image names
embeddings = np.load('embeddings.npy')
image_names = np.load('image_names.npy')

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # Using L2 distance
index.add(embeddings)  # Add all embeddings to the index

def find_similar_images(query_image, top_n=6):
    # Preprocess the query image
    query_image = preprocess_image(np.array(query_image))

    # Generate embedding for the query image
    query_embedding = model.predict(tf.expand_dims(query_image, axis=0))

    # Search for the top N most similar images
    D, I = index.search(query_embedding, top_n)

    # Retrieve the similar image filenames
    similar_images = [image_names[i] for i in I[0]]

    return similar_images

@app.route('/find_similar', methods=['POST'])
def find_similar():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    file = request.files['image']
    if file:
        # Open the uploaded image
        img = Image.open(file)

        # Find similar images
        similar_images = find_similar_images(img)

        # Return the result as a JSON response
        return jsonify({'similar_images': similar_images})

    return jsonify({'error': 'Invalid image file'}), 400



if __name__ == '__main__':
    app.run(debug=True,port=8000)

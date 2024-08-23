import tensorflow as tf
import numpy as np
from PIL import Image
import os
# Load a pretrained model (example, EfficientNet)
model = tf.keras.applications.EfficientNetB1(weights='imagenet', include_top=False, pooling='avg')


def preprocess_image(image):
    img = tf.image.resize(image, [224, 224])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


embeddings = []
image_names = []

for image_file in os.listdir('fashion_images/'):
    image = Image.open(f'fashion_images/{image_file}')
    image = preprocess_image(np.array(image))
    embedding = model.predict(tf.expand_dims(image, axis=0))
    embeddings.append(embedding)
    image_names.append(image_file)

# Save the embeddings and image names
np.save('embeddings.npy', np.vstack(embeddings))
np.save('image_names.npy', image_names)

import faiss

# Load embeddings and image names
embeddings = np.load('embeddings.npy')
image_names = np.load('image_names.npy')

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # Using L2 distance
index.add(embeddings)  # Add all embeddings to the index

def find_similar_images(query_image_path, top_n=6):
    # Load and preprocess the query image
    query_image = Image.open(query_image_path)
    query_image = preprocess_image(np.array(query_image))

    # Generate embedding for the query image
    query_embedding = model.predict(tf.expand_dims(query_image, axis=0))

    # Search for the top N most similar images
    D, I = index.search(query_embedding, top_n)

    # Retrieve the similar image filenames
    similar_images = [image_names[i] for i in I[0]]

    return similar_images

# Test the function
query_image_path = 'fashion_images/image_113.jpg'  # Path to your query image
similar_images = find_similar_images(query_image_path, top_n=6)
print("Similar Images:", similar_images)
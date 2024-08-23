import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np
import faiss
from torchvision import models
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


# Load a pretrained model (example, EfficientNet in PyTorch)
model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1)

model = nn.Sequential(*list(model.children())[:-1])  # Remove the classification head
model.eval()  # Set model to evaluation mode

# Define preprocessing transformations
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def preprocess_image(image):
    image = preprocess(image).unsqueeze(0)  # Add batch dimension
    return image


embeddings = []
image_names = []

for image_file in os.listdir('fashion_images/'):
    image = Image.open(f'fashion_images/{image_file}').convert('RGB')
    image = preprocess_image(image)

    # Compute the embedding
    with torch.no_grad():
        embedding = model(image).squeeze().numpy()  # Remove batch dimension
    embeddings.append(embedding)
    image_names.append(image_file)

# Save the embeddings and image names
np.save('embeddings.npy', np.vstack(embeddings))
np.save('image_names.npy', image_names)

# Load embeddings and image names
embeddings = np.load('embeddings.npy')
image_names = np.load('image_names.npy')

# Create FAISS index
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # Using L2 distance
index.add(embeddings)  # Add all embeddings to the index


def find_similar_images(query_image_path, top_n=6):
    # Load and preprocess the query image
    query_image = Image.open(query_image_path).convert('RGB')
    query_image = preprocess_image(query_image)

    # Generate embedding for the query image
    with torch.no_grad():
        query_embedding = model(query_image).squeeze().numpy()  # Remove batch dimension

    # Search for the top N most similar images
    D, I = index.search(np.expand_dims(query_embedding, axis=0), top_n)

    # Retrieve the similar image filenames
    similar_images = [image_names[i] for i in I[0]]

    return similar_images


# Test the function
query_image_path = 'fashion_images/image_1523.jpg'  # Path to your query image
similar_images = find_similar_images(query_image_path, top_n=6)
print("Similar Images:", similar_images)

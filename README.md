# Fashion Image Similarity API

This repository provides an API for finding similar fashion images based on a query image. The API allows you to upload an image, and it returns the names of six similar images from a pre-existing dataset of fashion images. The project leverages both TensorFlow and PyTorch for feature extraction and uses FAISS for efficient image similarity search.

## Project Structure

- **fashion_images/**: Contains 331 images used for similarity searches.
- **app.py**: Flask app that serves the API for uploading an image and getting the names of six similar images.
- **main.py**: TensorFlow-based implementation for extracting image features and building the FAISS index.
- **main(pytorch).py**: PyTorch-based implementation of the same functionality as `main.py`.
- **upload.html**: HTML form for uploading an image to the API.
- **colab_files/fashion_using_vgg.ipynb**: Google Colab notebook for running the project using VGG16 to extract image features and recommend similar fashion items.

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/fashion-image-similarity.git
   cd fashion-image-similarity
2. Install the required dependencies:
  ```bash
  pip install -r requirements.txt
 ```
3. Make sure you have the necessary pre-trained models and saved embeddings. If using the Colab notebook, the steps to connect to Google Drive and unzip the dataset are outlined in `fashion_using_vgg.ipynb`

## Running the API
1. To run the Flask app, use the following command:
   ```bash
   python app.py
2. Open your browser and navigate to `http://127.0.0.1:8000/`. You will see an upload form where you can select an image and submit it to find similar fashion items.

## Using the API
- Upload an image using the provided form in upload.html.
- The API will return a JSON response with the names of six images that are most similar to the uploaded image.

## TensorFlow and PyTorch Implementations
- `main.py` (TensorFlow) : Uses EfficientNetB1 to extract image embeddings and stores them in a FAISS index.
- `main(pytorch).py` (PyTorch) : An equivalent implementation of the TensorFlow code but using PyTorch.

## Google Colab Notebook
- For users who prefer to run the project using vgg in Google Colab, the notebook `fashion_using_vgg.ipynb` provides a complete workflow,(flask part is not covered here):
- Connect to Google Drive.
- Extract images from a ZIP file.
- Load VGG16 for feature extraction.
- Recommend similar fashion items based on a query image.

## Example Usage
### Finding Similar Images (API):
open `upload.html` file ,Upload an image:

![image](https://github.com/user-attachments/assets/08f4c0a0-8709-4cd3-a8b0-c0f2bac1ac5b)

## Example response:
![image](https://github.com/user-attachments/assets/26306495-b1e6-43c4-b71e-91c113f3ebb2)

## Finding Similar Images (Colab):
In the Colab notebook, load your dataset, preprocess images, and use the VGG16 model to recommend fashion items. The code displays similar images alongside the input image.

## Example Usage
```python
input_image_path = '/content/fashion_images_folder/fashion_images/image_113.jpg'
recommend_fashion_items_cnn(input_image_path, all_features, image_paths_list, model, top_n=4)
```
## Example response:
![image](https://github.com/user-attachments/assets/d377e19b-5a98-42a5-9f37-02f1d66c6723)

## Acknowledgements:
- This project uses pre-trained models from TensorFlow and PyTorch.
- FAISS is used for efficient similarity search.




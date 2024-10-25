import os
import zipfile
import torch
import joblib
from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms
from PIL import Image
import logging
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Constants for image dimensions
IMG_WIDTH = 160
IMG_HEIGHT = 160


# Set up logging configuration
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
# Initialize Flask app
app = Flask(__name__)

# Initialize MTCNN and InceptionResnetV1 models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Preprocessing transforms
preprocess = transforms.Compose(
    [
        transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.50, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)

# Global variables for classifier and label encoder
classifier = None
label_encoder = None

def load_dataset(dir):
    """Load dataset and extract embeddings and labels."""
    embeddings = []
    labels = []

    for sub_dir in os.listdir(dir):
        person_dir = os.path.join(dir, sub_dir)
        if not os.path.isdir(person_dir):
            continue

        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)

            ext_face = extract_face(image_path)
            if ext_face is None:
                continue  # Skip if no face was detected

            face_embd = get_embedding(ext_face)
            embeddings.append(face_embd.flatten())  # Flatten to avoid 2D arrays
            labels.append(sub_dir)  # Use sub_dir as the label instead of image_name

    return np.array(embeddings), np.array(labels)

def extract_face(image_path):
  image = Image.open(image_path).convert('RGB')
  image_tensor = preprocess(image)
  img_cropped = image_tensor.unsqueeze(0).to(device)
  return img_cropped

def get_embedding(img_cropped):
    """Get the embedding for a cropped face image."""
    with torch.no_grad():
        embedding = model(img_cropped)
    return embedding.cpu().numpy()

def model_train(X_train, y_train):
    """Train the SVM classifier and save it."""
    classifier = SVC(kernel='linear', probability=True)
    classifier.fit(X_train, y_train)
    # Save the trained model
    joblib.dump(classifier, 'svm_model.pkl')
    return classifier



def get_first_folder(directory_path):
    # List all items in the directory
    items = os.listdir(directory_path)
    # Filter to include only directories
    folders = [item for item in items if os.path.isdir(os.path.join(directory_path, item))]
    # Return the first folder if there are any, otherwise return None
    return folders[0] if folders else None




@app.route('/train', methods=['POST'])
def train():
    """Train the model with the provided dataset in a zip file."""
    global classifier, label_encoder

    # Ensure a file is provided
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
  
    # Check if a filename was given
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Create a directory for the extracted files
    extract_path = 'extracted_dataset'
    os.makedirs(extract_path, exist_ok=True)

    extract_dataset_path = 'extracted_dataset1'
    # Save the uploaded zip file temporarily
    zip_path = os.path.join(extract_path, file.filename)
    file.save(zip_path)

    # Extract the contents of the zip file
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dataset_path)
    except zipfile.BadZipFile:
        return jsonify({'error': 'Invalid zip file'}), 400

    logging.info(os.path.splitext(extract_dataset_path))
    logging.info('os.path.splitext(extract_dataset_path)[0]')

    # Load dataset from the extracted directory
    dataset_dir = os.path.join(extract_dataset_path, get_first_folder(extract_dataset_path))
    logging.info(dataset_dir)
    if not os.path.exists(dataset_dir):
        return jsonify({'error': 'Dataset directory not found after extraction',"main":dataset_dir}), 400

    # Load embeddings and labels
    X, Y = load_dataset(dataset_dir)

    # Encode labels using LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(Y)

    # Split dataset into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Train the model and save it
    classifier = model_train(X_train, y_train)

    return jsonify({'message': 'Model trained and saved successfully!'}), 200

@app.route('/predict', methods=['POST'])
def predict():
    """Predict the label of the uploaded image."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    image_path = os.path.join('uploads', file.filename)
    file.save(image_path)

    ext_face = extract_face(image_path)
    if ext_face is None:
        return jsonify({'error': 'No face detected'}), 400

    face_embd = get_embedding(ext_face)
    face_embd_flat = face_embd.flatten().reshape(1, -1)

    prediction = classifier.predict(face_embd_flat)
    predicted_label = label_encoder.inverse_transform(prediction)

    return jsonify({'label': predicted_label[0]}), 200

def load_trained_model():
    """Load the trained SVM model from disk."""
    global classifier
    if os.path.exists('svm_model.pkl'):
        classifier = joblib.load('svm_model.pkl')
    else:
        classifier = None

if __name__ == "__main__":
    # Load the trained model when the app starts
    load_trained_model()

    # Ensure the upload folder exists
    os.makedirs('uploads', exist_ok=True)

    app.run(host='0.0.0.0', port=5000,debug=True)

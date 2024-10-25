import os 
import zipfile
import torch
from facenet_pytorch import MTCNN,InceptionResnetV1
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

IMG_WIDTH = 160
IMG_HEIGHT = 160





def getFiles():
    extract_path = '/train'
    zip_path ='/archive (8).zip'
    os.makedirs(extract_path,exit_ok=True)
    with zipfile.ZipFile(zip_path,'r') as zip:
        zip.extractall(extract_path)
   



def extract_face_detail(img):
    image = Image.open(image_path).convert('RGB')
    image_tensor = preprocess(image)
    img_cropped = image_tensor.unsqueeze(0).to(device)
    return img_cropped




def get_embedding(img):
    with torch.no_grad():
        embedding=model(image_path)
    return embedding.cpu().numpy()


def load_dataset(dir):
    embeddings:[]
    labels:[]
    
    for sub_dir in os.listdir(dir):
        person_dir = os.path.join(dir,sub_dir)
        if not os.path.isdir(person_dir):
            continue
        
        for image_name in os.listdir(sub_dir):
            image_path = os.path.join(sub_dir,image_name)
            
            ext_face = extract_face_detail(image_path)
            face_embd=get_embedding(ext_face)
            
            embeddings.append(face_embd)
            
            labels.append(image_name)
            
            
    return numpy.array(embeddings).squeeze() , numpy.array(labels)



def modelTrain(X,Y,X_test,Y_test):
    classifier = SVC(kernel='linear',probability=True)
    classifier.fit(X,Y)

    y_pred = classifier.predict(X_test)
    
    accuracy = accuracy_score(y_pred,Y_test)
    return accuracy



    


            
def getPrediction():
    # creating the model 

    model = InceptionResnetV1(pretrain='vggface2').eval()
    device = torch.device('cuda' if torch.cuda.is_available () else 'cpu')
    model = model.to(device)
    
    preprocess = transforms(
        [
          transforms.resize(IMG_WIDTH,IMG_HEIGHT),
          transforms.ToTensor(),
          transforms.Normalize(mean=[0.50,0.5,0.5],std=[0.5,0.5,0.5])  
        ]
    )
    
    X,Y = load_dataset('/train/Celebrity Faces Dataset/')
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(Y)
    
    
    X_train,X_test,y_train,y_test = train_test_split(X,y_encoded,test_size=0.2,random_state=42)
    
    accuracy = modelTrain(X_train,y_train,X_test,y_test)
    
    
    
    
    
    
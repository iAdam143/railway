from flask import Flask, request, render_template
from torchvision import transforms as t
import torch.nn.functional as F
import torch
import numpy as np
import cv2
import pickle

app = Flask(__name__, template_folder='templates', static_folder='Static')


@app.route("/")
def index():
    return render_template('index.html')

with open("modelcpu.pkl", "rb") as file:
    model = pickle.load(file)
model.eval()

def pre_process_image(img):
    pre_processing_transforms = t.Compose([
        t.ToPILImage(),
        t.Resize(size=(512, 512)),
        t.RandomRotation(degrees=(-20,+20)),
        t.ToTensor(),
        t.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = pre_processing_transforms(img)
    img = torch.unsqueeze(img, 0)
    return img

def classify_image(img):
    with torch.no_grad():
        output = model(img)
    output = F.softmax(output, dim=1)
    class_name = ['Crown And Root Rot','Healthy Leaf', 'Leaf Rust', 'Wheat Loose Smut']
    ps = output.cpu().data.numpy().squeeze()
    pred_class = class_name[np.argmax(ps)]
    return pred_class

@app.route('/predict', methods=['POST'])
def predict():
   
    if request.method == 'POST':
        img = request.files['image'].read()
        np_img = np.frombuffer(img, np.uint8)
        opencv_img = cv2.imdecode(np_img, cv2.IMREAD_UNCHANGED)
        processed_img = pre_process_image(opencv_img)
        prediction = classify_image(processed_img)
        img = cv2.imencode('.jpg', opencv_img)[1].tobytes() 
        return render_template('index.html', result=prediction)
   
if __name__ == "__main__":
    app.run()

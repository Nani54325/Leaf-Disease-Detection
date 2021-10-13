from flask import Flask, render_template, request,jsonify
from keras.models import load_model
import cv2
import numpy as np
import base64
from PIL import Image
from keras.preprocessing import image
import numpy as np
import io
import re

img_size=100

app = Flask(__name__) 

model=load_model('model/AlexNetModel.hdf5')

label_dict={0:'Apple Scab', 1:'Apple Black Rot', 2:'Apple Cedar Rust',3:'Apple Healthy',4:'Blueberry Healthy',
            5:'Cherry Powdery Mildew',6:'Cherry Healthy',7:'Corn Gray Leaf Spot',8:'Corn Common Rust',
            9:'Corn Northern Leaf Blight',10:'Corn Healthy',11:'Grape Black Rot',12:'Grape Esca (Black Measles)',
            13:'Grape Leaf Blight (Isariopsis Leaf Spot)',14:'Grape Healthy',15:'Orange Haunglongbing (Citrus Greening)',
            16:'Peach Bacterial Spot',17:'Peach Healthy',18:'Pepper Bell Bacterial Spot',19:'Pepper Bell Healthy',
            20:'Potato Early Blight',21:'Potato Late Blight',22:'Potato Healthy',23:'Raspberry Healthy',24:'Soybean Healthy',
            25:'Squash Powdery Mildew',26:'Strawberry Leaf Scorch',27:'Strawberry Healthy',28:'Tomato Bacterial Spot',
            29:'Tomato Early Blight',30:'Tomato Late Blight',31:'Tomato Leaf Mold',32:'Tomato Septoria Leaf Spot',
            33:'Tomato Two Spotted Spider Mite',34:'Tomato Target Spot',35:'Tomato Yellow Leaf Curl Virus',
            36:'Tomato Mosaic Virus',37:'Tomato Healthy'
           }

def preprocess(img):

	img=np.array(img)

	if(img.ndim==3):
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	else:
		gray=img

	gray=gray/255
	resized=cv2.resize(gray,(img_size,img_size))
	reshaped=resized.reshape(1,img_size,img_size)
	return reshaped

@app.route("/")
def index():
	return(render_template("index.html"))

@app.route("/predict", methods=["POST"])
def predict():
	print('HERE')
	message = request.get_json(force=True)
	encoded = message['image']
	decoded = base64.b64decode(encoded)
	dataBytesIO=io.BytesIO(decoded)
	dataBytesIO.seek(0)
	image1 = Image.open(dataBytesIO)

	new_img=image1.resize((224,224))


	#new_img = image.load_img(image_path, target_size=(224, 224))

	img = image.img_to_array(new_img)
	img = np.expand_dims(img, axis=0)
	img = img/255
	#print("Following is our prediction:")
	prediction = model.predict(img)
	d = prediction.flatten()
	j = d.max()
	indexvalue=""
	for index,item in enumerate(d):
		if item == j:
			print(index)
			indexvalue=label_dict[index]

	#print(prediction,result,accuracy)

	response = {'prediction': {'result': indexvalue}}

	return jsonify(response)

app.run(debug=True)

#<img src="" id="img" crossorigin="anonymous" width="400" alt="Image preview...">
import tensorflow as tf  
import cv2
from tensorflow.keras.models import load_model

# load model
path = r"D:\development\ai\Lung\mymodel.h6"
photo =r"D:\development\ai\Lung\test.jpg"
model = load_model(path)

# analysis
import numpy as np
from tensorflow import keras
a,l,s,n=0,0,0,0

for i in range(1,242):
	photo =r"D:\development\ai\Lung\data\test ("+str(i)+").jpg"
	img = cv2.imread(photo)
	img_2 = cv2.resize(img,(224,224))
	img2 = np.array(img_2)
	img_2 = img_2/255
	img_2 = tf.expand_dims(img_2, 0)

	# predictions
	predicted = model.predict([img_2])
	predicted = np.argmax(predicted, axis=1)
	result = predicted[0]
	if result == 0:
		a=a+1
		#print("\nRes: a")
	elif result == 1:
		l=l+1
		#print("\nRes: l")
	elif result == 2:
		n=n+1
		#print("\nRes: n")
	elif result == 3:
		s=s+1
		#print("\nRes: s")

print("\nA:"+str(a)+"\nL:"+str(l)+"\nN:"+str(n)+"\nS:"+str(s))


    
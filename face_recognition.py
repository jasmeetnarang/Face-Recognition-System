from keras.preprocessing import image
from PIL import Image
from mtcnn.mtcnn import MTCNN
import cv2
import numpy as np
import os
import pickle

import pandas as pd


# create the detector, using default weights
detector = MTCNN()

def extract_face(filename, required_size=(50, 50)):
	# load image from file
	pixels = cv2.imread(filename)
	# detect faces in the image
	results = detector.detect_faces(pixels)
	# print(results)
	# extract the bounding box from the first face
	x1, y1, width, height = results[0]['box']
	x2, y2 = x1 + width, y1 + height
	# extract the face
	face = pixels[y1:y2, x1:x2]
	# resize pixels to the model size
	image = Image.fromarray(face)
	image = image.resize(required_size)
	face_array = np.asarray(image)

	return face_array

gallery = "face_images/gallery"
probes = "face_images/probe"

# get the labels and features for each image
def get_features(directory):
	no_classes = len(os.listdir(directory))
	print("Total classes: ",no_classes)

	labels = []
	features = []

	label = 0
	count_r = 0
	for i in os.listdir(directory):
		# count_c = 0
		loc = os.path.join(directory,i)
		for f in os.listdir(loc):
			filename = os.path.join(loc,f)
			print("filename: ", filename)
			try:
				face = extract_face(filename)
				gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
				flatten = gray.flatten()
				print("flatten: ", flatten)
				print("flat length: ", len(flatten))
				features.append(flatten.astype("float32"))
				count_r += 1
				print("features array: ", features)
				labels.append(label)
				print("labels: ",labels)
			except:
				print("No face detected, moving to next image")
		# count_c += 1
		label += 1


	features = np.asarray(features)
	features = pd.DataFrame(features)

	return features,labels

features, labels = get_features(gallery)
# save data as pickle files
with open('Train_dataset', 'wb') as handle:
	pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('Train_labels', 'wb') as handle:
	pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load data from pickle file
with open('Train_dataset', 'rb') as f:
	X_train = pickle.load(f)

with open('Train_labels', 'rb') as f:
	y_train = pickle.load(f)


features, labels = get_features(probes)
# save data as pickle files
with open('Test_dataset', 'wb') as handle:
	pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('Test_labels', 'wb') as handle:
	pickle.dump(labels, handle, protocol=pickle.HIGHEST_PROTOCOL)


# load data from pickle file
with open('Test_dataset', 'rb') as f:
	X_test = pickle.load(f)

with open('Test_labels', 'rb') as f:
	y_test = pickle.load(f)


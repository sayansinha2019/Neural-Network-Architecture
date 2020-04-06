from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import SGD
from keras.layers import Dense
from keras.utils import np_utils
from imutils import paths
from keras.models import load_model
import numpy as np
import argparse
import cv2
import os


def image_to_feature_vector(image,size=(32,32)):
	return cv2.resize(image,size).flatten()

ap=argparse.ArgumentParser()
ap.add_argument("-d", "--dataset",required=True,help="path to input dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model file")
args =vars(ap.parse_args())

print("[INFO] describing images.....")
imagepaths=list(paths.list_images(args["dataset"]))

data=[]
labels=[]
for(i,imagepath) in enumerate(imagepaths):
	image=cv2.imread(imagepath)
	label=imagepath.split(os.path.sep)[-1].split(".")[0]
	features=image_to_feature_vector(image)
	data.append(features)
	labels.append(label)
	
	if i>0 and i%1000==0:
		print("[INFO] processed {}/{}".format(i,len(imagepath)))

le=LabelEncoder()
labels=le.fit_transform(labels)
data=np.array(data)/255.0
labels=np_utils.to_categorical(labels,2)
print("[INFO] constructing training/testing split....")
(traindata,testdata,trainlabels,testlabels)=train_test_split(data,labels,test_size=0.25,random_state=42)


model = Sequential()
model.add(Dense(768,input_dim=3072,init="uniform",activation="relu"))
model.add(Dense(384,activation="relu" ,kernel_initializer="uniform"))
model.add(Dense(2))
model.add(Activation("softmax"))

print("[INFO] compiling model ....")
sgd=SGD(lr=0.01)
model.compile(loss="binary_crossentropy",optimizer=sgd, metrics=["accuracy"])
model.fit(traindata,trainlabels,epochs=50,batch_size=128,verbose=1)
print("[INFO] evaluating on testing set...")
(loss,accuracy)=model.evaluate(testdata,testlabels,batch_size=128,verbose=1)
print("[INFO] loss={:.4f}%".format(loss,accuracy*100))
print("[INFO] dumping arch and weights to file ....")
model.save('cnnCat2.h5', overwrite=True)
 


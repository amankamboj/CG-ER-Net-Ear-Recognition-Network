import os
import cv2
import h5py
import keras
import numpy as np
from keras import metrics
from keras.layers import Dense, Activation
from keras.models import Model,load_model
from keras.utils import to_categorical
#from keras.applications import vgg16
from keras.applications import resnet50


#Load the dataset
def load_data(path):
    X_train = []
    Y_train = []
    for sub in os.listdir(path):
        for img in os.listdir(os.path.join(path, sub)):
            pose = cv2.imread(os.path.join(os.path.join(path, sub), img))
            print(pose)
            pose = cv2.resize(pose, (224, 224))
            pose = resnet50.preprocess_input(pose)
            X_train.append(pose)
            print(len(X_train))
            Y_train.append(to_categorical(int(sub)-1, 405))
            print(len(Y_train))
    return np.asarray(X_train), np.asarray(Y_train)


train_path = './EarDataAllsubjects'
test_path = './EarDataAllsubjects'

X_train, Y_train = load_data(train_path)
X_test, Y_test = load_data(test_path)


print("Done Loading Data....Start Training")


#create the VGG model and change the last layer to have 100 classes insted of 1000
model_vgg16 = resnet50.ResNet50(input_shape=(224, 224, 3), weights='imagenet', include_top=True)

x = Dense(405, kernel_initializer='glorot_normal', name='new_predictions')(model_vgg16.layers[-2].output)
y = Activation('softmax')(x)
ear_vgg16 = Model(inputs=model_vgg16.input, outputs=y)

#I am fine-tuning the complete model, and not just the last layer
'''
for i in range(len(ear_vgg16.layers[:-2])):
    ear_vgg16.layers[i].trainable=False
'''
#ear_vgg16=load_model('ear_resnet.h5py')
print(ear_vgg16.summary())

ear_vgg16.compile(optimizer=keras.optimizers.SGD(learning_rate=0.01), loss='categorical_crossentropy', metrics=[metrics.categorical_accuracy])
history = ear_vgg16.fit(X_train, Y_train, batch_size=32, epochs=200, verbose=1, shuffle=True)

#ear_vgg16.save_weights('ear_vgg16.h5py')
ear_vgg16.save('./Resnet_Results/ear_resnet.h5py')

acc = history.history['categorical_accuracy']
loss = history.history['loss']

loss1=np.asarray(loss)	

acc1=np.asarray(acc)


np.savetxt('Resnet_Results/Loss.txt',loss1)
np.savetxt('Resnet_Results/Val_Loss.txt',acc1)
#val_loss = history.history['val_loss']
print('/nfinal accuracy',acc)
print('final loss/n',loss)

#compute the Test Accuracy
y_hat_test = ear_vgg16.predict(X_test, batch_size=32, verbose=1)
true = 0
for i in range(405):
    if np.argmax(y_hat_test[i]) == np.argmax(Y_test[i]):
        true+=1

print("Test Accuracy:", true/405)

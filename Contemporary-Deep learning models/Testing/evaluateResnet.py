import cv2
import numpy as np
from keras.applications.vgg19 import VGG19
from keras.models import Model,load_model
from keras.models import model_from_json
from keras.layers import Input
from sklearn.metrics.pairwise import cosine_similarity
import os
import tensorflow as tf
import matplotlib.pyplot as plt


graph = tf.get_default_graph()

#X = Input(shape=(400, 300, 3))
#classifier = VGG19(weights="imagenet",input_tensor=X)
#feature = classifier.output
classifier=load_model('./Resnet_Results/ear_resnet.h5py')
#classifier = VGG19('vgg19_weights.h5')
feature = classifier.get_layer('activation_28').output
classifier.summary()
feature_model = Model(classifier.input,feature)
exit()


#base_model = VGG19(weights='imagenet')
#model = Model(inputs=base_model.input, outputs=base_model.get_layer('block4_pool').output)

def model_predict(img1,img2):
    with graph.as_default():
        img2_representation = feature_model.predict(img2)
        img1_representation = feature_model.predict(img1)
        print('img1_representation.shape=',img1_representation.shape)

        #similarity = cosine_similarity(np.reshape(img1_representation, [1, 12 * 9 * 512]),
                                   #np.reshape(img2_representation, [1, 12 * 9 * 512]))
        dist = (img1_representation-img2_representation)**2
        score = np.sum(dist)
    # print(similarity)
    """
    if i < 10 and similarity*100 >= 60 and f.split('-')[0] == m.split('-')[0]:
        tp += 1
    if i < 10 and similarity*100 < 60 and f.split('-')[0] != m.split('-')[0]:
        fn += 1
    """
    print('done')
    return score


images = [[0 for x in range(3)] for y in range(60)]

path='USTB_DB1'
s=os.listdir(path)
s.sort()

poses = 3

#mages =[][]
for i in s :
    print('subject',i)
    for p in range(1,poses+1):
        print(path+'/' +str(i)+'/'+str(p) + '.BMP')
        current_pose = cv2.imread(path+'/'+str(i)+'/'+str(p)+'.BMP')
        current_pose=cv2.resize(current_pose,(224,224))
        current_pose=current_pose/255.
        #np.reshape(cv2.imread(path+'/' +str(i)+'/'+str(1) + '.BMP'), [1, 400, 300, 3])
        k=int(i)-1
        l=int(p)-1
        print('k',k)
        print('l',l)    
        images[k][l]=np.reshape(current_pose, [1,224, 224, 3])
        
    '''    
    images[i][0] =  np.reshape(cv2.imread(path+'/' +str(i)+'/'+str(1) + '.BMP'), [1, 400, 300, 3])
    images[i][1] =  np.reshape(cv2.imread(path+'/' +str(i)+'/'+str(2) + '.BMP'), [1, 400, 300, 3])
    images[i][2] =  np.reshape(cv2.imread(path+'/' +str(i)+'/'+str(3) + '.BMP'), [1, 400, 300, 3])
    '''
    '''
    images[i][0] =  np.reshape(cv2.imread('database2/' + '{0:0=2d}'.format(i+1) + '-1' + '.BMP'), [1, 400, 300, 3])
    print('database2/' + '{0:0=2d}'.format(i+1) + '-1' + '.BMP')    
    images[i][1] =  np.reshape(cv2.imread('database2/' + '{0:0=2d}'.format(i+1) + '-2' + '.BMP'), [1, 400, 300, 3])
    print('database2/' + '{0:0=2d}'.format(i+1) + '-2' + '.BMP')
    images[i][2] =  np.reshape(cv2.imread('database2/' + '{0:0=2d}'.format(i+1) + '-3' + '.BMP'), [1, 400, 300, 3])
    print('database2/' + '{0:0=2d}'.format(i+1) + '-3' + '.BMP')
    images[i][3] =  np.reshape(cv2.imread('database2/' + '{0:0=2d}'.format(i+1) + '-4' + '.BMP'), [1, 400, 300, 3])
    print('database2/' + '{0:0=2d}'.format(i+1) + '-4' + '.BMP')    
    '''
#print(images.shape[0])
print('subjects are loaded')
f = open('Resnet_USTB.txt', 'w')

subjects=60
posses=3
galarry_image=2
flag=1

print('prediction strated')
#f = open('VGG_USTB2.txt', 'w')
for s1 in range(0,subjects):
    print('s1=',s1)
    for p1 in range(galarry_image,posses):
        print('p1=',p1)
        for s2 in range(0,subjects):
            print('s2=',s2)
            for p2 in range(0,galarry_image):
                print('p2=',p2) 
                print('prediction strated')
                score = model_predict(images[s1][p1],images[s2][p2])
                print(str(s1+1)+'\t' +str(p1+1)+'\t'+str(s2+1)+'\t'+str(p2+1)+'\t'+str(flag)+'\t'+str(score)+'\n')
                flag=0
                if s1==s2:
                    flag=1    
                f.write(str(s1+1)+'\t' +str(p1+1)+'\t'+str(s2+1)+'\t'+str(p2+1)+'\t'+str(flag)+'\t'+str(score)+'\n')



'''
for i in range(0,59):
    for k in range(0,2):
        for l in range(0,2):
            print('subject=',i)
            if(k<l) :
                flag = 1
                score = model_predict(images[i][k], images[i][l])
                f.write(str(i+1)+'\t'+str(k+1)+'\t'+str(i+1)+'\t'+str(l+1)+'\t'+str(flag)+'\t'+str(1-score[0][0] )+'\n')
            else:
                flag = 0
                score = model_predict(images[i][k], images[i+1][l])
                f.write(str(i+1)+'\t'+str(k+1)+'\t'+str(i+2)+'\t'+str(l+1)+'\t'+str(flag)+'\t'+str(1-score[0][0] )+'\n')
'''




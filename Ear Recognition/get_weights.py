import tensorflow as tf
import os
import sys
import argparse
import facenet
import numpy as np
import cv2
from datetime import datetime

def main(args):
    with tf.Graph().as_default():

        with tf.Session() as sess:

            learning_rate_placeholder = tf.placeholder(tf.float32, name='learning_rate')

            phase_train_placeholder = tf.placeholder(tf.bool, name='phase_train')

            image_placeholder = tf.placeholder(tf.float32, shape=(None,450,300,3), name='image')

            dynamic_alpha_placeholder = tf.placeholder(tf.float32, shape=(), name='dynamic_alpha_placeholder')

            input_map = {'image': image_placeholder, 'phase_train': phase_train_placeholder, 'learning_rate': learning_rate_placeholder, 'dynamic_alpha_placeholder': dynamic_alpha_placeholder}
            facenet.load_model(args.model, input_map=input_map)
            # Get output tensor
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

            for var in tf.trainable_variables():
                print(var)
            exit()
            
            subjects=106
            embeddings_array=[]
            for s in range(1,10):
                subject_emb = []
                #poses = len(os.listdir(args.data_dir+'/s'+str(s)))
                poses = 10
                for p in range(1,poses+1):
                    print(s,p)
                    current_pose = cv2.imread(args.data_dir+'/00'+str(s)+'/'+str(p)+'.jpg')
                    current_pose=cv2.resize(current_pose,(300,450))
                    current_pose=current_pose/255.

                    current_pose = np.expand_dims(current_pose, axis=0)
                    emb = sess.run(embeddings,feed_dict={image_placeholder: current_pose,phase_train_placeholder: False})
                    # np.save('/home/terone/Dr_Aditya_Nigam/Spatial_Unwarping/embeddings/s'+str(s)+'/'+str(p),emb.reshape((args.embedding_size,)))
                    subject_emb.append(emb.reshape((args.embedding_size,)))
                subject_emb = np.asarray(subject_emb)
                embeddings_array.append(subject_emb)
            for s in range(10,100):
                subject_emb = []
                poses = 10
                for p in range(1,poses+1):
                    print(s,p)
                    current_pose = cv2.imread(args.data_dir+'/0'+str(s)+'/'+str(p)+'.jpg')
                     
                    current_pose=cv2.resize(current_pose,(300,450))
                    current_pose=current_pose/255.

                    current_pose = np.expand_dims(current_pose, axis=0)
                    # feed_dict = {image_placeholder: current_pose}
                    emb = sess.run(embeddings,feed_dict={image_placeholder: current_pose,phase_train_placeholder: True})
                    # np.save('/home/terone/Dr_Aditya_Nigam/Spatial_Unwarping/embeddings/s'+str(s)+'/'+str(p),emb.reshape((args.embedding_size,)))
                    subject_emb.append(emb.reshape((args.embedding_size,)))
                subject_emb = np.asarray(subject_emb)
                embeddings_array.append(subject_emb)
            for s in range(100,107):
                subject_emb = []
                poses = 10
                for p in range(1,poses+1):
                    print(s,p)
                    current_pose = cv2.imread(args.data_dir+'/'+str(s)+'/'+str(p)+'.jpg')
                    current_pose=cv2.resize(current_pose,(300,450))
                    current_pose=current_pose/255.
                    

                    current_pose = np.expand_dims(current_pose, axis=0)
                    # feed_dict = {image_placeholder: current_pose}
                    emb = sess.run(embeddings,feed_dict={image_placeholder: current_pose,phase_train_placeholder: True})
                    # np.save('/home/terone/Dr_Aditya_Nigam/Spatial_Unwarping/embeddings/s'+str(s)+'/'+str(p),emb.reshape((args.embedding_size,)))
                    subject_emb.append(emb.reshape((args.embedding_size,)))
                subject_emb = np.asarray(subject_emb)

            # embeddings_array=[]
                embeddings_array.append(subject_emb)
            # for s in range(1,subjects+1):
            #     subject_emb = []
            #     poses = len(os.listdir(args.data_dir+'/s'+str(s)))
            #     for p in range(1,poses+1):
            #         current_file = np.load('/home/terone/Dr_Aditya_Nigam/Spatial_Unwarping/embeddings/s'+str(s)+'/'+str(p)+'.npy')
            #         subject_emb.append(current_file)
            #     subject_emb = np.asarray(subject_emb)
            #     embeddings_array.append(subject_emb)
            print('embeddings calculated')
            subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')
            gallery_images=args.number_of_gallery_images
            f = open('ear'+subdir+'.txt', 'w')
            for s1 in range(subjects):
                poses1 = embeddings_array[s1].shape[0]
                for p1 in range(gallery_images,poses1):
                # for p1 in range(gallery_images):
                    for s2 in range(subjects):
                        poses2 = embeddings_array[s1].shape[0]
                        for p2 in range(gallery_images):
                            emb1 = embeddings_array[s1][p1,:]
                            emb2 = embeddings_array[s2][p2,:]
                            dist = (emb1-emb2)**2
                            score = np.sum(dist)
                            flag = 0
                            if s1==s2:
                                flag=1
                            f.write(str(s1+1)+'\t'+str(p1+1)+'\t'+str(s2+1)+'\t'+str(p2+1)+'\t'+str(flag)+'\t'+str(score)+'\n')

            


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str,
        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file',default='./models/siamese/Ear')
    parser.add_argument('--data_dir', type=str,
        help='Path to the data directory',default='/mnt/Data2/Aman_Kamboj/7_pt_rotated_rect1')
    parser.add_argument('--number_of_gallery_images', type=int,
        help='Number of images used for training for each subject',default=5)
    parser.add_argument('--embedding_size', type=int,
        help='Dimensionality of the embedding.', default=128)
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

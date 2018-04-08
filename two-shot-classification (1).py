
# coding: utf-8

# In[1]:


# two-shot-classification.py
# The training set will have two coppies of each exaple. I will run single test 
#file against the tarining examples each time. 
import os
import numpy as np
from scipy.ndimage import imread
from scipy.spatial.distance import cdist
import tensorflow as tf
import itertools 


# In[9]:


nrun = 20  # Number of test files
#path_to_script_dir = os.path.dirname(os.path.realpath(__file__))
path_to_script_dir='C:/Users/Esta/Projects/two-shots'
path_to_all_runs = os.path.join(path_to_script_dir, 'all_runs')
fname_label = 'class_labels.txt'  # Where class labels are stored for each run


# In[5]:


#I merge the training examples of earch two follders in allrun to 
#create a two coppies for each training. I will use the test examples of first folder


def classification_run(folder1, folder2, f_load, f_cost, alpha):
    # Compute error rate for one run of two-shot classification
    # Input
    #  folder1 : contains images for a run of one-shot classification
    #  folder2 : contains the images in the following folder for a run of one-shot classification

    #  f_load : f_load('file.png')=load_img_as_points('file.png') should read in the image file and
    #           process it
    #  f_cost : f_cost(image1, image2, image3, margin) should compute similarity between two
    #           images using the triple loss
    #  alpha  : it is the margin that we use in L2 distance.
    #
    # Output
    #  perror : percent errors (0 to 100% error)
    
    with open(os.path.join(path_to_all_runs, folder1, fname_label)) as f:
        pairs1 = [line.split() for line in f.readlines()]


    with open(os.path.join(path_to_all_runs, folder2, fname_label)) as k:
        pairs2 = [line.split() for line in k.readlines()]
        
    
    # Unzip the pairs into two sets of tuples
    test_files1, train_files1 = zip(*pairs1)
    test_files2, train_files2 = zip(*pairs2)
   

    answers_files = list(train_files1)  # Copy the training file list
    test_files1 = sorted(test_files1)
    train_files1 = sorted(train_files1)

    answers_files2 = list(train_files2)  # Copy the training file list
    test_files2 = sorted(test_files2)
    train_files2 = sorted(train_files2)
    
    train_files=train_files1 + train_files2

    n_train = len(train_files)
    n_test1 = len(test_files1)
    n_test2 = len(test_files2)
    


    # Load the images (and, if needed, extract features)
    # we are using the function 'load_img_as_points(filename)' to convert to image
    
    train_items = [f_load(os.path.join(path_to_all_runs, f)) 
                   for f in train_files]
    test_items1 = [f_load(os.path.join(path_to_all_runs, f))
                   for f in test_files1]

    answer_items1 = [f_load(os.path.join(path_to_all_runs, f))
                   for f in answers_files1] #I will use it as a positive examples in L_2
  
    

    # Compute cost matrix
    costM = np.zeros((n_test1, n_train))#create a two dim matrix


            
    for i, test_i in enumerate(test_items1): # test_i and train_j are images points  
        for j, train_j in enumerate(train_items):
            costM[i, j] = f_cost(test_i, train_j, answer_items1[i], alpha)#calcula the distance using "L2_distance()
                
    y_hats = np.min(costM, axis=1) 
   

    #compute the error rate by counting the number of correct predictions
    correct = len([1 for y_hat in y_hats
                  if y_hat <= 0.01])#i choose the threshhold=0.01
              

    pcorrect = correct / float(n_test1)  # ensure float division
    perror = (1.0 - pcorrect)*100

    
    return perror


# In[6]:



def L2_distance(test, negative, positive, margin):
    # Modified L2_distance
    # Input
    # test= is the test examples
    #negative= images in the training examples
    #positive= this is the result from the one-shot-training 
    # we add a margine to the inequality of the triple loss
    # output : \\f(A)-f(B)\\^2 - \\f(A)-f(C)\\^2 + alpha.
    D_pos = cdist(test, positive)
    D_neg = cdist(test, negative)
    D_pos_sqr=np.square(D_pos)
    D_neg_sqr=np.square(D_neg)
    D_pos_sum=np.sum(D_pos_sqr)
    D_neg_sum=np.sum(D_neg_sqr)
    D_sub=float(np.add(np.subtract(D_neg_sum, D_pos_sum), margin))
    loss=np.maximum(D_sub , 0.0 )
    return loss 


# In[7]:



def load_img_as_points(filename):
    # Load image file and return coordinates of black pixels in the binary image
    #
    # Input
    #  filename : string, absolute path to image
    #
   
    I = imread(filename, flatten=True)
    # Convert to boolean array and invert the pixel values
    I = ~np.array(I, dtype=np.bool)
    # Create a new array of all the non-zero element coordinates
    D = np.array(I.nonzero()).T

    return D - D.mean(axis=0)


# In[10]:



# Main function
if __name__ == "__main__":

 # The result will use the test examples in 10 files from allruns
    perror = np.zeros(10)
    margine=0.4

    k=0 
    for r in range(0, 20, 2):
        perror[k]= classification_run('run{:02d}'.format(r+1), 'run{:02d}'.format(r + 2),
                                       load_img_as_points,
                                       L2_distance, margine)
                          
        print(' run {:02d} (error {:.1f}%)'.format(r, perror[k]))
        k=k+1
        

    total = np.mean(perror)
    print('Average error {:.1f}%'.format(total))


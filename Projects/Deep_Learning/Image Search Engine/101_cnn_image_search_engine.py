
# we want to pass in our 300 images to the vgg ntwork
# we waant to be returned an ouput from the global average pooling layer
# different from our max pooling layer 
# featyer array or vector of 512 numbers 
# create an object of 312 numbers 
# we compare feature to base 


#########################################################################
# Convolutional Neural Network - Image Search Engine
#########################################################################


###########################################################################################
# import packages
###########################################################################################

from tensorflow.keras.models import Model,load_model
from tensorflow.keras.applications.vgg16 import VGG16,preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from os import listdir
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import pickle


###########################################################################################
# bring in pre-trained model (excluding top)
###########################################################################################

# image parameters
# we are only interested in the features it lerned along the way 

img_width = 224
img_height = 224
num_channels = 3
# network architecture

vgg = VGG16(input_shape=(img_width,img_height,num_channels),
            include_top=False,
            pooling='avg') # single set of numbers to represent all of those features 
# global avg pooling will be applied to that final layer thus output of model will be one array than many

vgg.summary()

model = Model(inputs = vgg.inputs,
              outputs = vgg.layers[-1].output)# our output the one that we want is the final layer of vgg 

# save model file
save_dir = 'C:/Users/eacalder/Documents/GitHub/DataScienceInfinity/Saved_files/Deep_Learning/Image_Search_Engine/models/vgg16_search_.h5'
model.save(save_dir)

###########################################################################################
# preprocessing & featurising functions
###########################################################################################

# image pre-processing function

def preprocess_image(filepath):
    image = load_img(filepath,
                 target_size=(img_width,img_height))
    image = img_to_array(image)
    image = np.expand_dims(image,axis=0)
    image = preprocess_input(image) #scale 
    return image
# featurise image

def featureise_image(image):
    feature_vector = model.predict(image)
    return feature_vector

###########################################################################################
# featurise base images
###########################################################################################

# source directory for base images
dir = 'C:/Users/eacalder/Documents/GitHub/DataScienceInfinity/Saved_files/Deep_Learning/Image_Search_Engine/'

source_dir = 'C:/Users/eacalder/Documents/GitHub/DataScienceInfinity/Saved_files/Deep_Learning/Image_Search_Engine/data/'

# empty objects to append to
filename_store = [] #store to retrieve if necessary 
feature_vector_store = np.empty((0,512))

# pass in & featurise base image set
for image in listdir(source_dir):
    print(image)
    # append image filename for future lookup
    filename_store.append(source_dir + image)

    # preprocess the image 
    preprocessed_image = preprocess_image(source_dir + image)

    # extract the feature vector
    feature_vector = featureise_image(preprocessed_image)

    #append feature vector for similarity calculations 
    feature_vector_store = np.append(feature_vector_store,
                                     feature_vector,
                                     axis = 0)

# save key objects for future use

pickle.dump(filename_store,open(dir + 'models/filename_store.p', 'wb'))
pickle.dump(feature_vector_store,open(dir + 'models/feature_vector_store.p', 'wb'))

        
###########################################################################################
# pass in new image, and return similar images
###########################################################################################

# load in required objects

model = load_model(save_dir, 
                   compile = False)# avoid a warning message, compile is for traiing which we arent doing

filename_store = pickle.load(open(dir + 'models/filename_store.p','rb'))
feature_vector_store = pickle.load(open(dir + 'models/feature_vector_store.p','rb'))

# search parameters
search_results_n = 8 # 8 closest images will be retrend
# search_image = dir + 'search_image_01.jpg'
search_image = dir + 'search_image_02.jpg'
        
# preprocess & featurise search image
preprocessed_image = preprocess_image(search_image)
search_feature_vector = featureise_image(preprocessed_image)

        
# instantiate nearest neighbours logic
image_neighbors =  NearestNeighbors(n_neighbors=search_results_n,
                                    metric='cosine')


# apply to our feature vector store
image_neighbors.fit(feature_vector_store)

# return search results for search image (distances & indices)

image_distances, image_indices = image_neighbors.kneighbors(search_feature_vector)
print(image_distances)
print(image_indices)

# convert closest image indices & distances to lists

image_indices = list(image_indices[0])
image_distances = list(image_distances[0])
print(image_distances)
print(image_indices)
# get list of filenames for search results
search_results_file = [filename_store[i] for i in image_indices]
print(search_results_file)

# plot results

plt.figure(figsize=(12,9))
for counter, result_file in enumerate(search_results_file): # axes counter in which image is in make it look nice   
    image = load_img(result_file)
    ax = plt.subplot(3, 3, counter+1) # one plot with a 3 by 3 
    plt.imshow(image)
    plt.text(0, -5, round(image_distances[counter],3)) # see cosine distance score 
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()






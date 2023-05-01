#########################################################################
# Convolutional Neural Network - Fruit Classification
#########################################################################


#########################################################################
# Import required packages
#########################################################################

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, Flatten,Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
#########################################################################
# Set Up flow For Training & Validation data
#########################################################################

# data flow parameters
training_data_dir = "C:/Users/eacalder/Documents/GitHub/DataScienceInfinity/Saved_files/Deep_Learning/CNN/data/training"
validation_data_dir = "C:/Users/eacalder/Documents/GitHub/DataScienceInfinity/Saved_files/Deep_Learning/CNN/data/validation"

batch_size = 32
img_width = 224 # vgg16 used this instead of 128
img_height = 224 # vgg16 used this instead of 128
num_channels = 3
num_classes = 6
# we are restricted by our local pcs ram 

# image generators
# powerful conept. generators bring in to rescale. we need to standerdize 
# similarly like we do before. 

training_generator = ImageDataGenerator(preprocessing_function = preprocess_input, #rescale=1./255, due to vgg16
                                        rotation_range=20,
                                        width_shift_range=0.2,
                                        height_shift_range=0.2,
                                        zoom_range=0.1,
                                        horizontal_flip=True,
                                        brightness_range=(0.5,1.5),
                                        fill_mode='nearest',
                                        ) # 0-255 to 0-1
validation_generator = ImageDataGenerator(preprocessing_function = preprocess_input ) #rescale=1./255) # 0-255 to 0-1


# image flows
training_set = training_generator.flow_from_directory(directory=training_data_dir,
                                                      target_size=(img_width,img_height),
                                                      batch_size= batch_size,
                                                      class_mode='categorical'
                                                      )

validation_set = validation_generator.flow_from_directory(directory=validation_data_dir,
                                                      target_size=(img_width,img_height),
                                                      batch_size= batch_size,
                                                      class_mode='categorical'
                                                      )

#########################################################################
# Network Architecture
#########################################################################
input_shape=(img_width,img_height,num_channels)
# network architecture

vgg = VGG16(input_shape=(img_width,img_height,num_channels),
            include_top=False)# very important we specified 
# remember only the initial convolutional layers not the dense. since it is massive 

vgg.summary()
# we dont wan t it to be updated, freeze all layers they wont be updated duringtraining 

for layer in vgg.layers:
    layer.trainable = False
vgg.summary()

flatten = Flatten()(vgg.output)

# relu chosen for vgg16
dense1 = Dense(128, # 128 nuerons 
               activation='relu')(flatten)

dense2 = Dense(128, 
               activation='relu')(dense1)

#output layer will be dense 
output = Dense(num_classes, 
               activation='softmax')(dense2)

model = Model(inputs = vgg.inputs,
              outputs = output)

# compile network
model.compile(loss= 'categorical_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])

# warnings related to a gpu

# view network architecture

model.summary()

#########################################################################
# Train Our Network!
#########################################################################

# training parameters
num_epochs = 10 # 50 times before we are done, who knows if too much refine from there 
model_filename = 'C:/Users/eacalder/Documents/GitHub/DataScienceInfinity/Saved_files/Deep_Learning/CNN/models/fruits_cnn_vgg.h5' #model architecture and values for network params


# callbacks
save_best_model = ModelCheckpoint(filepath=model_filename,
                                  monitor='val_accuracy',
                                  mode='max',
                                  verbose=1,
                                  save_best_only=True)


# train the network
history = model.fit(x = training_set,
                    validation_data = validation_set,
                    batch_size=batch_size,
                    epochs = num_epochs,
                    callbacks=[save_best_model])

# 
#########################################################################
# Visualise Training & Validation Performance
#########################################################################

import matplotlib.pyplot as plt

# plot validation results
fig, ax = plt.subplots(2, 1, figsize=(15,15))
ax[0].set_title('Loss')
ax[0].plot(history.epoch, history.history["loss"], label="Training Loss")
ax[0].plot(history.epoch, history.history["val_loss"], label="Validation Loss")
ax[1].set_title('Accuracy')
ax[1].plot(history.epoch, history.history["accuracy"], label="Training Accuracy")
ax[1].plot(history.epoch, history.history["val_accuracy"], label="Validation Accuracy")
ax[0].legend()
ax[1].legend()
plt.show()

# get best epoch performance for validation accuracy
max(history.history['val_accuracy'])

# we might want to set our epoch down to 10-20 as it 
# flat lines going further, best performance setting to lower 
# the major issue here is that we see some crazy over fitting 
# our network is learning so well that it vant make a judgement with slightly different things 
# cant generalize well
# remember epoch 50, filter 32, 2 convolutional layers, dense layer with 32 neurons  

#########################################################################
# Make Predictions On New Data (Test Set)
#########################################################################

# import required packages
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import pandas as pd
from os import listdir

# parameters for prediction

model_filename = 'C:/Users/eacalder/Documents/GitHub/DataScienceInfinity/Saved_files/Deep_Learning/CNN/models/fruits_cnn_vgg.h5' #model architecture and values for network params
img_width = 224
img_height = 224
labels_list = ['apple','avocado', 'banana', 'kiwi', 'lemon','orange']

# load model
model = load_model(model_filename)

# image pre-processing function

def preprocess_image(filepath):
    image = load_img(filepath,
                 target_size=(img_width,img_height))
    print(image)
    image = img_to_array(image)
    print(image)
    print(image.shape)
    image = np.expand_dims(image,axis=0)
    print(image.shape)
    image = preprocess_input(image) #scale 
    return image
# image prediction function

def make_prediction(image):
    class_probs = model.predict(image)
    print(class_probs)

    predicted_class = np.argmax(class_probs)
    print(predicted_class)

    predicted_label = labels_list[predicted_class]
    print(predicted_label)

    predicted_prob = class_probs[0][predicted_class]
    print(predicted_prob)
    return predicted_label, predicted_prob

# image = preprocess_image(filepath)
# make_prediction(image)# clever tweaks to our network to make better predictions below

# here i think is the best way to evaluate  manually doing this

# loop through test data
source_dir = 'C:/Users/eacalder/Documents/GitHub/DataScienceInfinity/Saved_files/Deep_Learning/CNN/data/test/'
folder_names = ['apple','avocado', 'banana', 'kiwi', 'lemon','orange']
actual_labels = []
predicted_labels = []
predicted_probabilities = []
filenames = []

# we are doing this more manually

for folder in folder_names:
    images = listdir(source_dir + '/' + folder) 
    for image in images:
        processed_image = preprocess_image(source_dir + '/' + folder + '/' + image)
        predicted_label, predicted_prob = make_prediction(processed_image)

        actual_labels.append(folder) 
        predicted_labels.append(predicted_label)
        predicted_probabilities.append(predicted_prob)
        filenames.append(image)

# create dataframe to analyse
predictions_df = pd.DataFrame({'actual_label': actual_labels,
                               'predicted_labels':predicted_labels,
                               'predicted_probability':predicted_probabilities,
                               'filenames':filenames})
print(predictions_df.info())
print(predictions_df.shape)
print(predictions_df.head())

# im more interested in the ones that were wrongly predicted 

predictions_df['correct'] = np.where(predictions_df['actual_label'] == predictions_df['predicted_labels'], 1,0)
print(predictions_df.info())
print(predictions_df.shape)
print(predictions_df.head())

# overall test set accuracy

test_set_accuracy = predictions_df['correct'].sum()/len(predictions_df)
print(test_set_accuracy)
# 81% basic 
# 83% dropout
# 95% image aug.
# 92% vgg. maybe try epoch higher 
# confusion matrix (raw)

# maybe predicting well with apple but not with lemons 

confusion_matrix = pd.crosstab(predictions_df['predicted_labels'],predictions_df['actual_label'])
print(confusion_matrix)

# confusion matrix (percentages )

confusion_matrix = pd.crosstab(predictions_df['predicted_labels'],predictions_df['actual_label'], normalize = 'columns')
print(confusion_matrix)

predictions_df.sort_values(by=['predicted_probability'])






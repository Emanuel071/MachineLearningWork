
#########################################################################
# Artificial Neural Network - Video Game Success Prediction
#########################################################################


#########################################################################
# Import Libraries
#########################################################################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.callbacks import ModelCheckpoint
#########################################################################
# Import Data
#########################################################################

# import data
data_path = "C:/Users/eacalder/Documents/GitHub/DataScienceInfinity/Saved_files/Deep_Learning/ANN/data/ann-game-data.csv"

data_for_model = pd.read_csv(data_path)
print(data_for_model.info())
print(data_for_model.shape)
print(data_for_model.head())

# scaling here will help us ensure the training model process 
# unscaling will provide errneous values
# player id will be removed 
# we have to hot encode to ensure we are givning model numeric data
# drop any redundant columns
data_for_model.drop("player_id", axis = 1, inplace = True)
print(data_for_model.info())
print(data_for_model.shape)
print(data_for_model.head())

#########################################################################
# Split Input Variables & Output Variable
#########################################################################

X = data_for_model.drop(["success"], axis = 1)
y = data_for_model["success"]
print(X.shape)
print(y.shape)

#########################################################################
# Split out Training & Test sets
#########################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)
# 80% go to our training set and the remainder will go to our test set 
# stratify = y to ensure we get an eveen balance of the output class
#########################################################################
# Deal with Categorical Variables
#########################################################################

categorical_vars = ["clan"]

one_hot_encoder = OneHotEncoder(sparse=False, drop = "first")

X_train_encoded = one_hot_encoder.fit_transform(X_train[categorical_vars])
X_test_encoded = one_hot_encoder.transform(X_test[categorical_vars])

encoder_feature_names = one_hot_encoder.get_feature_names_out(categorical_vars)

X_train_encoded = pd.DataFrame(X_train_encoded, columns = encoder_feature_names)
X_train = pd.concat([X_train.reset_index(drop=True), X_train_encoded.reset_index(drop=True)], axis = 1)
X_train.drop(categorical_vars, axis = 1, inplace = True)

X_test_encoded = pd.DataFrame(X_test_encoded, columns = encoder_feature_names)
X_test = pd.concat([X_test.reset_index(drop=True), X_test_encoded.reset_index(drop=True)], axis = 1)
X_test.drop(categorical_vars, axis = 1, inplace = True)

print(X_test.info())
print(X_test.shape)
print(X_test.head())

print(X_train.info())
print(X_train.shape)
print(X_train.head())

# think there was 3 human ork and elf and it makes sense we did human and ork with 
# no elf column as we know that if we did we would run into the issue of over fitting
# we specified drop = first to avoid multicollenearity. but keep in mind for nueral networks
# this is not the case and would not hurt the model. we could have left it in but willperform 
# the same
#########################################################################
# Feature Scaling
#########################################################################

scale_norm = MinMaxScaler()
X_train = pd.DataFrame(scale_norm.fit_transform(X_train), columns = X_train.columns)
X_test = pd.DataFrame(scale_norm.transform(X_test), columns = X_test.columns)

print(X_train.info())
print(X_train.shape)
print(X_train.head())

# we see all of the variables range between 0 and 1 

#########################################################################
# Network Architecture
#########################################################################

# network architecture
model = Sequential()

model.add(Dense(units=32,  #32 nuerons
                input_dim = X_train.shape[1],
                )) 
model.add(Activation('relu')) # most popular, computationally light, good performance 
# we want this to be a deep nueral network by definition 2 layers 

# keras knows the inputs for 2nd hidden layer will be output of 1st prolly 
# due to sequential 
model.add(Dense(units=32
                )) 
model.add(Activation('relu'))

# 2 is probably enough

# we want to return a probability so no relu
model.add(Dense(units=1
                )) 
model.add(Activation('sigmoid')) # if multi class we would use soft max, and for regression linear

# compile network
model.compile(loss='binary_crossentropy',# for multi class categorical cross entropy, regression MSE
              optimizer='adam',
              metrics=['accuracy']
              ) 


# view network architecture
model.summary()
# Total params: 1,345
# Trainable params: 1,345
# Non-trainable params: 0

#########################################################################
# Train Our Network!
#########################################################################

# training parameters

num_epoch = 50 # we dont know if too many or enough, try something and see
batch_size = 32 # again we dont know but common to start
model_file_name = 'C:/Users/eacalder/Documents/GitHub/DataScienceInfinity/Saved_files/Deep_Learning/ANN/models/video_game_ann.h5'
# save our model into our model folder
# h5 is something keras uses 

# callbacks
# remember we will save the best model 

save_best_model = ModelCheckpoint(filepath = model_file_name,
                                  monitor='val_accuracy',
                                  mode = 'max',
                                  verbose=1, # default is 0
                                  save_best_only=True
                                  )

# keras lingo callback, early stopppings, monitoring training, train for 1000 epochs
# we can do on the fly training, we can as the learning rate be lowered, we could be stuck in 
# a local minima that keras could help, we can log the data in a csv file 
# one we will use is model checkpoint. save model for future use during training 
# if we se improvements, keras will update the model that is in harddrive if improvement is seen

# accuracy is good when is high and loss is good when it is low



# train the network
history = model.fit(x=X_train.values, # X is DF we need to pass as series instead
                    y= y_train, # already panda series
                    validation_data= (X_test,y_test),
                    batch_size=batch_size,
                    epochs=num_epoch,
                    callbacks=[save_best_model], # list, updates the file we save 
                    )

# neural networks work in a stochastic way or random 
#########################################################################
# Visualise Training & Validation Performance
#########################################################################

import matplotlib.pyplot as plt

# plot metrics by epoch
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

# best performance at about 30 or 40 epochs
# isnt much of a gap between two lines which is good, if it wasnt 
# then we would have a overfitting issue, learning so well it is struggling 
# cant generalize. we are risking performance on new data. if we were facing 
# we can do drop out. where we force our network to randomly ignore a proportion of the nuerons 
# in hidden layer. no neuron can be too hard wired 
#  

#########################################################################
# Make Predictions On New Data
#########################################################################

# import packages

from tensorflow.keras.models import load_model

# load model

model = load_model(model_file_name )

# create new data

list(X_train)

player_a = [[9,30,6,11, 62, 0,1]]
player_a = scale_norm.transform(player_a) 
print(player_a)

# make our prediction

prediction = model.predict(player_a)
print(prediction)
# 11% chance of succeeding 


player_b = [[11,27,0,9, 59, 0,0]]
player_b = scale_norm.transform(player_b) 
print(player_b)

# make our prediction

prediction = model.predict(player_b)
print(prediction)
# 94% chance of succeeding 

# if instead we wnated to give a stake holders a pass or fail 

prediction_class = (prediction > 0.5) * 1
print(prediction_class)
 




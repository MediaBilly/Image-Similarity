# Code executable for part A
import keras
import numpy as np
import argparse
import os
from imageDataset import ImageDataset, createDataset
from utility import *
from keras import Model, Input, optimizers, layers
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from encoder import encoder
from decoder import decoder
from math import ceil,sqrt

# Initialize GPU
init_gpu()

# Parse command line arguments
args_parser = argparse.ArgumentParser()
args_parser.add_argument('-d', '--Dataset')
args_parser.add_argument('-q', '--Queryset')
args_parser.add_argument('-od','--Output_Dataset_File')
#args_parser.add_argument('-oq','--Output_Query_File')
args = args_parser.parse_args()

dataset_file = args.Dataset
queryset_file = args.Queryset
output_dataset_file = args.Output_Dataset_File
#output_query_file = args.Output_Query_File

# Check if dataset file exists
if os.path.isfile(dataset_file):
    # Read Dataset
    dataset = ImageDataset(dataset_file)

    # Input type construction
    x_dimension, y_dimension = dataset.getImageDimensions()
    inChannel = 1
    input_img = Input(shape=(x_dimension, y_dimension, inChannel))

    # Load images from dataset and normalize their pixels in range [0,1]
    images_normed = dataset.getImagesNormalized()

    # Split dataset into train and validation datasets
    X_train, X_validation, y_train, y_validation = train_test_split(
        images_normed,
        images_normed,
        test_size=0.2,
        random_state=13
    )
    
    # User Arguments
    convolutional_layers = int(input("Number of convolutional layers: "))
    convolutional_filter_size = int(input("Convolutional filter size: "))
    
    convolutional_filters_per_layer = []
    for layer in range(convolutional_layers):
        convolutional_filters_per_layer.append(int(input("Convolutional filters of layer " + str(layer + 1) + ": ")))
    
    epochs = int(input("Epochs: "))
    batch_size = int(input("Batch size: "))
    embedding_dimension = 10

    # Clear previous layer session to prevent saving same depth layers with different names
    K.clear_session()

    # Build the autoencoder
    fc_size = (int(dataset.getImageDimensions()[0] / 4)) * int((dataset.getImageDimensions()[1] / 4)) * convolutional_filters_per_layer[-1]
    encoded = encoder(input_img, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer, 0)
    flatten = layers.Flatten()(encoded)
    embedding = layers.Dense(embedding_dimension,name='embedding')(flatten)
    fc = layers.Dense(fc_size)(embedding)
    reshape = layers.Reshape(((int(dataset.getImageDimensions()[0] / 4)),(int(dataset.getImageDimensions()[1] / 4)),convolutional_filters_per_layer[-1]))(fc)
    decoded = decoder(reshape, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer, 0)
    
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop())

    # Print it's summary
    autoencoder.summary()

    # Train the autoencoder
    autoencoder_train = autoencoder.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(X_validation, y_validation)
    )
    
    # Save experiment results for later use
    '''
    parameters = {
        "Convolutional layers": convolutional_layers,
        "Convolutional filter size": convolutional_filter_size,
        "Convolutional filters per layer": convolutional_filters_per_layer,
        "Batch size": batch_size
    }
    '''

    # Read queryset
    queryset = ImageDataset(queryset_file)
    query_images = queryset.getImagesNormalized()

    # Get latent vectors for all query images
    latent_vector_model = Model(inputs=autoencoder.input,outputs=autoencoder.get_layer('embedding').output)
    latent_vectors = latent_vector_model.predict(queryset.getImagesNormalized())

    # Transform them in range {0,1,...,25500}
    scaler = MinMaxScaler(feature_range=(0,25500))
    scaler.fit(latent_vectors)
    latent_vectors = scaler.transform(latent_vectors)
    latent_vectors = np.array(latent_vectors)
    latent_vectors = np.around(latent_vectors).astype(int).tolist()
        
    # Save latent vectors in output_dataset_file
    createDataset(latent_vectors, output_dataset_file)
        
        
    # Save trained model weights
    if get_user_answer_boolean("Save trained model (Y/N)? "):
        save_file_path = input("Input save file path: ")
        autoencoder.save_weights(save_file_path)
        
else:
    print("Could not find dataset file: " + dataset_file)
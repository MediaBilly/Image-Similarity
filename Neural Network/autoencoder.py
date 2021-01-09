# Code executable for part A
import keras
import numpy as np
import argparse
import os
from experiment import Experiment
from imageDataset import ImageDataset
from utility import *
from keras import Model, Input, optimizers
from keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from encoder import encoder
from decoder import decoder

# Initialize GPU
init_gpu()

repeat = True
# Parse command line arguments
args_parser = argparse.ArgumentParser()
args_parser.add_argument('-d', '--Dataset')
args = args_parser.parse_args()

dataset_file = args.Dataset

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
    
    experiments = list()
    
    while repeat:
        # User Arguments
        convolutional_layers = int(input("Number of convolutional layers: "))
        convolutional_filter_size = int(input("Convolutional filter size: "))
        
        convolutional_filters_per_layer = []
        for layer in range(convolutional_layers):
            convolutional_filters_per_layer.append(int(input("Convolutional filters of layer " + str(layer + 1) + ": ")))
        
        epochs = int(input("Epochs: "))
        batch_size = int(input("Batch size: "))

        # Clear previous layer session to prevent saving same depth layers with different names
        K.clear_session()

        # Build the autoencoder
        encoded = encoder(input_img, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer, 0)
        decoded = decoder(encoded, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer, 0)
        
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(),metrics=['accuracy'])

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
        parameters = {
            "Convolutional layers": convolutional_layers,
            "Convolutional filter size": convolutional_filter_size,
            "Convolutional filters per layer": convolutional_filters_per_layer,
            "Batch size": batch_size
        }

        experiments.append(Experiment(parameters, autoencoder_train.history))
        
        # Prompt to plot experiments
        if get_user_answer_boolean("Show loss graphs (Y/N)? "):
            # Generate plots for all experiment losses
            for index, experiment in enumerate(experiments):
                fig = plt.subplot(len(experiments), 1, index + 1)
                experiment.generate_plot()

        plt.show()
            
        # Save trained model weights
        if get_user_answer_boolean("Save trained model (Y/N)? "):
            save_file_path = input("Input save file path: ")
            autoencoder.save_weights(save_file_path)
        
        repeat = get_user_answer_boolean("Repeat Experiment (Y/N)? ")
        
else:
    print("Could not find dataset file: " + dataset_file)

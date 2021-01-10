# Code executable for part B
import argparse
import os
import numpy as np
from imageDataset import ImageDataset
from labelDataset import LabelDataset
from encoder import encoder, encoder_layers
from keras import layers, Input, Model, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from keras.utils import to_categorical
from keras.losses import categorical_crossentropy
from keras import backend as K
from experiment import Experiment
from utility import *
from matplotlib import pyplot as plt
from math import ceil, sqrt


def save_clustering_result(dataset, classifier, filename):
    train_images = dataset.getImagesNormalized()
    y_pred = classifier.predict(train_images, batch_size=batch_size)
    predicted_classes = np.argmax(np.round(y_pred), axis=1)
    
    clusters = [list() for i in range(10)]

    for index, prediction in enumerate(predicted_classes):
        clusters[prediction].append(index)

    if os.path.exists(filename):
        os.remove(filename)

    clustering_output = open(filename,"a")
    for i in range(len(clusters)):
        clustering_output.write("CLUSTER-{} {{ size: {}".format(i,len(clusters[i])))
        for img_index in clusters[i]:
            clustering_output.write(", {}".format(img_index))
        clustering_output.write("}}\n")
    
    clustering_output.close()

# Initialize GPU
init_gpu()

# Parse command line arguments
args_parser = argparse.ArgumentParser()
args_parser.add_argument('-d',    '--training_set')
args_parser.add_argument('-dl',   '--training_labels')
args_parser.add_argument('-t',    '--testset')
args_parser.add_argument('-tl',   '--test_labels')
args_parser.add_argument('-model','--autoencoder_h5')
args = args_parser.parse_args()

training_set_file = args.training_set
training_labels_file = args.training_labels
testset_file = args.testset
test_labels_file = args.test_labels
autoencoder_weights_file = args.autoencoder_h5

# Read training set
training_set = ImageDataset(training_set_file)

# Read training labels
training_labels = LabelDataset(training_labels_file)

# Read testset
testset = ImageDataset(testset_file)

# Read test labels
test_labels = LabelDataset(test_labels_file)


experiments = list()
repeat = True

# Input type construction
x_dimension, y_dimension = training_set.getImageDimensions()
inChannel = 1
input_img = Input(shape=(x_dimension, y_dimension, inChannel))

# Split dataset into train and validation datasets
X_train, X_validation, y_train, y_validation = train_test_split(
    training_set.getImagesNormalized(),
    to_categorical(training_labels.get_labels(),num_classes=training_labels.num_classes()),
    test_size=0.2,
    random_state=13
)

while repeat:
    # User Arguments
    convolutional_layers = int(input("Number of convolutional layers: "))
    convolutional_filter_size = int(input("Convolutional filter size: "))
    
    convolutional_filters_per_layer = []
    for layer in range(convolutional_layers):
        convolutional_filters_per_layer.append(int(input("Convolutional filters of layer " + str(layer + 1) + ": ")))
    
    fully_connected_size = int(input("Fully connected layer size: "))
    epochs_phase_1 = int(input("Phase 1 epochs: "))
    epochs_phase_2 = int(input("Phase 2 epochs: "))
    batch_size = int(input("Batch size: "))
    embedding_dimension = int(input("Embedding Dimension: "))
    dropout_rate = float(input("Dropout rate: "))

    # Clear previous layer session to match continiously constructed layer names in weights file
    K.clear_session()

    # Construct the classifier NN(input -> encoder -> Flatten -> FC -> output with 10 classes(0 - 9))
    fc_size = (int(training_set.getImageDimensions()[0] / 2 ** min(convolutional_layers,2))) * int((training_set.getImageDimensions()[1] / 2 ** min(convolutional_layers,2))) * convolutional_filters_per_layer[-1]
    encoded = encoder(input_img, convolutional_layers, convolutional_filter_size, convolutional_filters_per_layer, dropout_rate)
    flatten = layers.Flatten()(encoded)
    embedding = layers.Dense(embedding_dimension,name='embedding')(flatten)
    fc = layers.Dense(fully_connected_size,name="fully_connected")(embedding)
    dropout = layers.Dropout(rate=dropout_rate,name="final_dropout")(fc)
    output_layer = layers.Dense(training_labels.num_classes(), activation='softmax',name="output")(dropout)

    classifier = Model(input_img, output_layer)
    classifier.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(),metrics=['accuracy'])

    # Print it's summary
    classifier.summary()

    # Load encoder weights
    classifier.load_weights(autoencoder_weights_file, by_name=True)

    # Train phase 1: Only fully connected layer weights

    # Make encoder layers non trainable
    for layer in classifier.layers[0 : encoder_layers(convolutional_layers)]:
        layer.trainable = False

    classifier_trained_phase1 = classifier.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs_phase_1,
        verbose=1,
        validation_data=(X_validation, y_validation)
    )

    # Train phase 2: All layer weights

    # Make encoder layers trainable
    for layer in classifier.layers[0:encoder_layers(convolutional_layers)]:
        layer.trainable = True

    classifier_trained_phase2 = classifier.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs_phase_2,
        verbose=1,
        validation_data=(X_validation, y_validation)
    )

    history = {
        'loss': classifier_trained_phase1.history['loss'] + classifier_trained_phase2.history['loss'],
        'val_loss': classifier_trained_phase1.history['val_loss'] + classifier_trained_phase2.history['val_loss'],
    }

    # Save experiment results for later use
    parameters = {
        "Fully connected size": fully_connected_size,
        "Dropout rate": dropout_rate,
        "Batch size": batch_size
    }

    experiments.append(Experiment(parameters, history))

    # Prompt to plot experiments
    if get_user_answer_boolean("Show loss graphs (Y/N)? "):
        # Generate plots for all experiment losses
        for index, experiment in enumerate(experiments):
            fig = plt.subplot(len(experiments), 1, index + 1)
            experiment.generate_plot()

    plt.show()
    
    # Save clustering results (K=10 classes)
    save_clustering_result(testset, classifier,"clustering_test_set.txt")
    save_clustering_result(training_set, classifier,"clustering_train_set.txt")

    # Prompt to predict test set
    if get_user_answer_boolean("Classify test set (Y/N)? "):
        # Predict test images
        test_images = testset.getImagesNormalized()
        y_pred = classifier.predict(test_images, batch_size=batch_size)
        predicted_classes = np.argmax(np.round(y_pred), axis=1)
        true_labels = test_labels.get_labels()
        target_names = [str(i) for i in range(test_labels.num_classes())]

        # Evaluate test set
        test_evaluation = classifier.evaluate(test_images,to_categorical(true_labels,num_classes=test_labels.num_classes()),verbose=0)
        print('Test loss: ', test_evaluation[0])
        print('Test accuracy: ', accuracy_score(true_labels, predicted_classes))
        
        # Print corrent and incorrect labels
        correct_labels = 0
        incorrect_labels = 0
        for index, label in enumerate(true_labels):
            if label == predicted_classes[index]:
                correct_labels += 1
            else:
                incorrect_labels += 1
        
        print('Found ' + str(correct_labels) + ' correct labels')
        print('Found ' + str(incorrect_labels) + ' incorrect labels')

        # Classification report
        print(classification_report(true_labels, predicted_classes, target_names=target_names))

        # Print confusion matrix
        print(confusion_matrix(true_labels, predicted_classes))


        # Show 100 of the classified test images with their predicted class 
        fig = plt.figure()
        fig_size = 100
        row_size = column_size = ceil(sqrt(fig_size))
        for index, img in enumerate(test_images[:min(100, len(test_images))]):
            fig.add_subplot(row_size, column_size, index + 1)
            plt.title(str(predicted_classes[index]))
            plt.imshow(img * 255)
            
        plt.show()

    repeat = get_user_answer_boolean("Repeat Experiment (Y/N)? ")

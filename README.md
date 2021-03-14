# Image Similarity

> The purpose of this project is to represent the hand-written diigit images from the MNIST dataset in lower dimension(reduced space) vectors using a Convolutional Autoencoder Neural Network with a dense layer(bottleneck). Then, the goal is to perform nearest neighbour search using the bruteforce method in the reduced space vectors, and compare the performance and accuracy with the LSH and Bruteforce Search in the original space with dimension 28x28=784. The other goal is to perform clustering using k-Means initialization and Lloyd's assignment in the original and reduced space and compare the performance and accuracy(using average silhouette) between the 2 spaces and also the clustering that the Neural Network Classifer does. We also implemented the Earth Movers Distance metric for image similarity.

> In order to run the programs in parts B,C,D you first need to compile them. To do so, navigate to `NearestNeighboursAndClustering` folder and run `make`. To remove object and executable folders, run `make clean` in the same folder.

## A. Convolutional Autoencoder with bottleneck layer for dimensionality reduction
Usage: Navigate to `NeuralNetwork` folder and then run as: `python reduce.py -d <dataset> -q <queryset> -od <output_dataset_file> -oq <output_query_file>`\
The program will first prompt the user if he either wants to load a pre-trained autoencoder model or train one from scratch. In this case, he will be prompted to enter the hyperparameters for the model. After the training is done, he will be prompted to either show the loss graphs, save the last trained model, save the latent representations in the MNIST binary format or repeat the training experiment with other hyperparameters.

### Required Libraries:
* tensorflow
* keras
* numpy
* sklearn
* matplotlib
* It is also recommended to have cuda installed for faster training

## B. Nearest Neighbour Search using Bruteforce Algorithm, LSH, and Bruteforce Algorithm in reduced space.
Usage: Navigate to `NearestNeighboursAndClustering` folder and then run as: `./search -d <input file original space> -i <input file new space> -q <query file original space> -s <query file new space> -k <int> -L <int> -ο <output file>`

#### Some Benchmarks:
| Latent Dimension | LSH Approximation Factor | Reduced Approximation Factor | Time(seconds) |
| :--------------: |:-----:                   | :----------:                 |:-------------:|
|        10        |          1.11199         | 1.56615                      |129.271        |
|        15        |          1.11199         | 1.45668                      |146.854        |
|        20        |          1.11199         | 1.35298                      |181.147        |

In the above benchmarks, the approximation factor is the average ratio of the distance to the Nearest Neighbour found using the approximation factor to the distance to the Nearest Neighbour found using Bruteforce Search in the original space. So the closest to one, the better the approximation is.

## C. Earth Movers Distance Metric
This program implements the Earth Movers Distance metric between 2 MNIST images, finds the 10 nearest neighbours for each o the query images with both metrics and calculates the correctness of the 2 metrics. The correctness is defined as the rate of the nearest neighbours that have the same label with the query image, so the bigger the ratio, the better the accuracy. We did experiments dividing the images in clusters with sizes 14x14, 7x7 and 4x4, but the final program divides the images in 7x7 clusters.\
Usage: Navigate to `NearestNeighboursAndClustering` folder and then run as: `./search -d  <input  file  original  space> -q  <query  file  original  space>  -l1  <labels of input dataset> -l2 <labels of query dataset> -ο <output file> -EMD`

#### Some Benchmarks:
| Cluster size | EMD Correctness | Mannhattan Correctness | Time |
| :--------------:  |:-----:                   | :----------:          |:-------------:|
|       14x14       |          0.08         | 0.78                     |20 minutes     |
|        7x7        |          0.47         | 0.78                     |1 hour         |
|        4x4        |          0.81         | 0.78                     |6 hours        |

## D. Clustering in Original Space, Reduced Space and using the Neural Network Classifier
Usage: Navigate to `NearestNeighboursAndClustering` folder and then run as: `./cluster –d <input file original space> -i <input file new space>  -n <classes from NN as clusters file> –c <configuration file> -o <output file>`\
In order for this to run, you first need to run the Neural Network Classifier to obtain the clustering results from that. To do so,navigate to `NeuralNetwork` folder and run it as: `python classification.py -d <training set> -dl <training labels> -t <testset> -tl <test labels> -model <autoencoder h5>` where in the model argument, you should give the Autoencoder Model from the first part.

#### Some Benchmarks:
| Clustering Space | Average Silhouette | Objective Function | Clustering Time(seconds) |
| :--------------:  |:-----:            | :----------:          |:-------------:|
|       NEW SPACE   |   0.0790247       | 1.93026e+08           |0.722248       |
| ORIGINAL SPACE    |   0.097787        | 1.89179e+08           |14.4419        |
|CLASSES AS CLUSTERS|   0.102485        | 1.93981e+08           |-              |

## Contributors:
1. [Vasilis Kiriakopoulos](https://github.com/MediaBilly)
2. [Dimitris Koutsakis](https://github.com/koutsd)
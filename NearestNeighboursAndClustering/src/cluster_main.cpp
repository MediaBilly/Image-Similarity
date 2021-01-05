#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <ctime>
#include <sstream>
#include "../headers/dataset.h"
#include "../headers/cluster.h"
#include "../headers/utilities.h"

void usage() {
    std::cout << "Usage:./cluster -d <input file original space> -i <input file new space> -n <classes from NN as clusters file> -c <configuration file> -o <output file>\n";
}

template<typename PixelType>
void silhouette(std::vector<Image<PixelType>*> &images, std::vector<Cluster<PixelType>*> &clusters, std::unordered_map<int, Cluster<PixelType>*> &clusterHistory, std::ofstream &outputStream) {
    // Calculate Silhouette for all images
    double averageSilhouette = 0.0;
    std::vector<double> s;
    for (unsigned int i = 0; i < images.size(); i++) {
        // Calculate distance of ith image to all the clusters
        Cluster<PixelType> *neighbourCluster = NULL, *closestCluster = NULL;
        double distToClosestCentroid = 1.0/0.0, distToSecondClosest = 1.0/0.0;
        
        for (unsigned int j = 0; j < clusters.size(); j++) {
            double dist = images[i]->distance(clusters[j]->getCentroid(), 1);

            if (dist < distToClosestCentroid) {
                distToSecondClosest = distToClosestCentroid;
                distToClosestCentroid = dist;

                neighbourCluster = closestCluster;
                closestCluster = clusters[j];
            }
            else if(dist < distToSecondClosest) {
                distToSecondClosest = dist;
                neighbourCluster = clusters[j];
            }
        }
        // Calculate average distance of ith image to images in same cluster
        double ai = clusterHistory[images[i]->getId()]->avgDistance(images[i]);
        // Calculate average distance of ith image to images in the next best(neighbor) cluster
        double bi = neighbourCluster->avgDistance(images[i]);
        // Calculate Silhouette for ith image
        double si = (bi - ai)/std::max(ai, bi);
        s.push_back(si);
        averageSilhouette += si;
    }
    // Print Silhouettes
    outputStream << "Silhouette: [";

    // Calculate and print average Silhouette for each cluster
    for (unsigned int i = 0; i < clusters.size(); i++) {
        double avgS = 0.0;
        std::vector<Image<PixelType>*> clusterPoints = clusters[i]->getPoints();
        for (unsigned int j = 0; j < clusterPoints.size(); j++) {
            avgS += s[clusterPoints[j]->getId()];
        }
        outputStream << avgS/clusterPoints.size() << ", ";
    }
    
    // Print average Silhouette for all points in dataset
    averageSilhouette /= images.size();
    outputStream  << averageSilhouette << "]\n";
}

template<typename PixelType>
double objectiveFunction(std::vector<Image<PixelType>*> &images, std::vector<Cluster<PixelType>*> &clusters) {
    double obj = 0.0;

    for (unsigned int i = 0; i < images.size(); i++) {
        double minD = 1.0/0.0;

        for (unsigned int j = 0; j < clusters.size(); j++) {
            double d = images[i]->distance(clusters[j]->getCentroid());
            if (d < minD) {
                minD = d;
            }
        }
        obj += minD;
    }

    return obj;
}

template<typename PixelType>
void cluster(Dataset<PixelType> *dataset, int K, std::ofstream &outputStream) {
    // Get images from dataset
    std::vector<Image<PixelType>*> images = dataset->getImages();

    // Initialize uniform random distribution number generator
    std::default_random_engine generator;
    std::uniform_int_distribution<int> uniform_distribution(1,images.size());

    // k-Means++ initialization:
    // Choose a centroid uniformly at random (indexing ranges from 1 to n)
    std::unordered_set<int> centroids;
    centroids.insert(uniform_distribution(generator));
    for (int t = 1; t < K; t++) {
        // For all non-centroids i, let D(i) = min distance to some centroid, among t chosen centroids and calculate P(r) = sum{D(i), 0 <= i <= r}
        double *P = new double[images.size() - t + 1];
        int *non_cendroid_index = new int[images.size() - t + 1];
        P[0] = 0;
        // Calculate max{D(i)} for all non-centroids i
        unsigned long long maxDi = 0;
        for (unsigned int i = 1,j = 0; j < images.size(); j++) {
            // Check if jth point is not a centroid and if so, keep it's index , calculate D(i) and use it to calculate P(i) using prefix sum technique. Otherwise, continue to next point.
            if (centroids.find(j+1) == centroids.end()) {
                // j is not a centroid
                // Compute D(i)
                double D = 1.0/0.0;
                for (auto c : centroids) {
                    double dist = images[c-1]->distance(images[i-1],1);
                    if (dist < D) {
                        D = dist;
                    }
                }
                i++;
                if (D >= maxDi) {
                    maxDi = D;
                }
            }
        }
        // i is 1-starting index for all non-centriods and j is 0-starting index for all points
        for (unsigned int i = 1,j = 0; j < images.size(); j++) {
            // Check if jth point is not a centroid and if so, keep it's index , calculate D(i) and use it to calculate P(i) using prefix sum technique. Otherwise, continue to next point.
            if (centroids.find(j+1) == centroids.end()) {
                // j is not a centroid
                // Compute D(i)
                double D = 1.0/0.0;
                for (auto c : centroids) {
                    double dist = images[c-1]->distance(images[i-1],1);
                    if (dist < D) {
                        D = dist;
                    }
                }
                D /= maxDi;
                P[i] = P[i-1] + D * D;
                non_cendroid_index[i] = j+1;
                i++;
            }
        }
        
        // Choose new centroid: r chosen with probability proportional to D(r)^2
        std::uniform_real_distribution<float> floatDistribution(0,P[images.size() - t]);

        // Pick a uniformly distributed float x ∈ [0,P(n−t)] and return r ∈ {1,2,...,n−t} : P(r−1) < x ≤ P(r), where P(0) = 0.
        float x = floatDistribution(generator);
        int left = 1,right = images.size() - t,r = 0;
        // Find r using binary search to P
        while (left <= right) {
            r = (left+right)/2;
            if (P[r-1] < x && x <= P[r]) { // P[r-1] < x <= P[r]
                break;
            }
            else if (x <= P[r-1]) {  // x <= P[r-1] <=  P[r]
                right = r - 1;
            }
            else {  // P[r-1] <= P[r] < x
                left = r + 1;
            }
        }
        // Add chosen centroid r to centroids set
        centroids.insert(non_cendroid_index[r]);

        delete[] non_cendroid_index;
        delete[] P;
    }
    // Initialize clusters for all centroids
    std::vector<Cluster<PixelType>*> clusters;
    unsigned int cid = 0;
    for (auto c : centroids) {
        clusters.push_back(new Cluster<PixelType>(*images[c-1],cid++));
    }

    unsigned int assignments;
    std::unordered_map<int, Cluster<PixelType>*> clusterHistory;

    // Clustering time!!!
    clock_t begin_clustering_time = clock();
    do {
        for (unsigned int i = 0; i < clusters.size(); i++) {
            clusters[i]->clear();
        }
        assignments = 0;
        // Assignment step
        // Lloyd's algorithn
        for (unsigned int i = 0; i < images.size(); i++) {
            // Find closest cluster for the current(ith) image
            double minDist = 1.0/0.0;
            Cluster<PixelType> *minCluster = NULL;
            for (unsigned j = 0; j < clusters.size(); j++) {
                double dist = images[i]->distance(clusters[j]->getCentroid(),1);
                if (dist < minDist) {
                    minDist = dist;
                    minCluster = clusters[j];
                }
            }
            // Insert the ith image to it's closest cluster
            minCluster->addPoint(images[i]);
            if (clusterHistory.find(images[i]->getId()) == clusterHistory.end() || clusterHistory[images[i]->getId()]->getId() != minCluster->getId()) {
                assignments++;
            }
            clusterHistory[images[i]->getId()] = minCluster;
        }

        // Update all cluster centroids
        if (assignments > 0) {
            for (unsigned int i = 0; i < clusters.size(); i++) {
                clusters[i]->updateCentroid();
            }
        }

    } while (assignments > 100);

    double clustering_time = double(clock() - begin_clustering_time) / CLOCKS_PER_SEC;

    // Print stats
    for (unsigned int i = 0; i < clusters.size(); i++) {
        outputStream << "CLUSTER-" << i+1 << " {size: " << clusters[i]->getSize() << ", centroid: [";
        for (int j = 0; j < dataset->getImageDimension() - 1; j++) {
            outputStream << (int)clusters[i]->getCentroid()->getPixel(j) << ", ";
        }
        outputStream << (int)clusters[i]->getCentroid()->getPixel(dataset->getImageDimension()-1) << "]}\n";
    }

    // Print clustering time
    outputStream << "clustering_time: " << clustering_time << std::endl;

    // Calculate Silhouette for all images
    silhouette<PixelType>(images, clusters, clusterHistory, outputStream);

    // Calculate Value of Objective Function: 
    outputStream << "Value of Objective Function: " << objectiveFunction(images, clusters) << std::endl;

    for (unsigned int i = 0;i < clusters.size();i++) {
        delete clusters[i];
    }
}


int main(int argc, char const *argv[]) {
    std::string nnClustersFile, configFile, outputFile, inputFileOriginalSpace, inputFileNewSpace;

    // Read command line arguments
    if (argc == 11) {
        for(int i = 1; i < argc; i+=2) {
            std::string arg(argv[i]);

            if(!arg.compare("-i")) {
                inputFileNewSpace = argv[i+1];
            }
            else if(!arg.compare("-c")) {
				configFile = argv[i+1];
			}
			else if(!arg.compare("-o")) {
				outputFile = argv[i+1];
			}
			else if(!arg.compare("-d")) {
				inputFileOriginalSpace = argv[i+1];
			}
            else if(!arg.compare("-n")) {
				nnClustersFile = argv[i+1];
			}
			else {
				usage();
				return 0;
			}
        }
    } 
    else {
        usage();
        return 0;
    }

    int K = 10; 

    std::ifstream config_ifs(configFile);

    std::string var;
    int value;
    
    // Read configuration file
    while(config_ifs >> var >> value) {
        if(var == "number_of_clusters:") {
            K = value;
        } else {
            std::cout << "Invalid configuration file" << std::endl;
            return 0;
        }
    }

    config_ifs.close();

    std::ofstream outputStream(outputFile);

    bool repeat;
    do {
        // Read Dataset
        Dataset<Pixel8Bit> *datasetOriginalSpace = new Dataset<Pixel8Bit>(inputFileOriginalSpace);
        Dataset<Pixel16Bit> *datasetNewSpace = new Dataset<Pixel16Bit>(inputFileNewSpace);

        if (datasetOriginalSpace->isValid() && datasetNewSpace->isValid()) {
            // Cluster new space images
            outputStream << "NEW SPACE" << std::endl;
            cluster<Pixel16Bit>(datasetNewSpace, K, outputStream);

            outputStream << std::endl;

            // Cluster original space images
            outputStream << "ORIGINAL SPACE" << std::endl;
            cluster<Pixel8Bit>(datasetOriginalSpace, K, outputStream);

            outputStream << std::endl;

            // Classes as clusters
            std::vector<Image<Pixel8Bit>*> images = datasetOriginalSpace->getImages();
            std::vector<Cluster<Pixel8Bit>*> clusters;
            std::unordered_map<int, Cluster<Pixel8Bit>*> clusterHistory;

            // Read NN clusters file
            std::ifstream clusters_ifs(nnClustersFile);
            std::string line;
            int cluster_index = 0;

            // Read each line corresponding to a cluster
            while (std::getline(clusters_ifs, line)) {
                std::istringstream iss(line);
                std::string skip, s;

                // Skip unecessary characters
                iss >> skip >> skip >> skip >> s;

                // Read cluster size
                //int clusterSize = get_num_from_string(s);

                // Initialize cluster
                clusters.push_back(new Cluster<Pixel8Bit>(cluster_index,datasetOriginalSpace->getImageWidth(),datasetOriginalSpace->getImageHeight()));

                // Add the images to it
                while(iss >> s) {
                    int imgIndex = get_num_from_string(s);
                    clusters[cluster_index]->addPoint(images[imgIndex]);
                    clusterHistory[imgIndex] = clusters[cluster_index];
                }

                // Update it's centroid to match it's images
                clusters[cluster_index]->updateCentroid();

                cluster_index++;
            }

            clusters_ifs.close();

            // Evaluation
            outputStream << "CLASSES AS CLUSTERS" << std::endl;
            silhouette<Pixel8Bit>(images,clusters,clusterHistory,outputStream);
            outputStream << "Value of Objective Function: " << objectiveFunction(images, clusters) << std::endl;

            // Delete memory allocated for nn clusters
            for (unsigned int i = 0;i < clusters.size();i++) {
                delete clusters[i];
            }
        }
        
        delete datasetNewSpace;
        delete datasetOriginalSpace;

        std::string promptAnswer;
        std::cout << "Do you want to enter another query (Y/N)?  ";

        do {
            std::cin >> promptAnswer;
        } while(promptAnswer != "Y" && promptAnswer != "N" && std::cout << "Invalid answer" << std::endl);

        repeat = (promptAnswer == "Y");

        if(repeat) {
            std::cout << "Input File original space: ";
            std::cin >> inputFileOriginalSpace;

            std::cout << "Input File new space: ";
            std::cin >> inputFileNewSpace;
            
            std::cout << "Classes from NN as clusters file: ";
            std::cin >> nnClustersFile;
        }

    } while(repeat);

    outputStream.close();

    return 0;
}

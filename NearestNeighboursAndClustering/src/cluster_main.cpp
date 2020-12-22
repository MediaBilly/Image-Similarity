#include <fstream>
#include <iostream>
#include <string>
#include <random>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <utility>
#include <ctime>
#include "../headers/dataset.h"
#include "../headers/cluster.h"

void usage() {
    std::cout << "Usage:./cluster -d <input file original space> -i <input file new space> -n <classes from NN as clusters file> -c <configuration file> -o <output file>\n";
}

template<typename Pixel8Bit>
double minDistBetweenClusterCentroids(std::vector<Cluster<Pixel8Bit>*> &clusters) {
    // Calculate min distance between centers to use it as start radius for range search in reverse assignment
    double ret = 1.0/0.0;
    for(unsigned int i = 0; i < clusters.size(); i++) {
        for(unsigned int j = i + 1; j < clusters.size(); j++) {
            double dist = clusters[i]->getCentroid()->distance(clusters[j]->getCentroid(), 1);

            if(dist < ret)
                ret = dist;
        }
    }
    return ret;
}

int main(int argc, char const *argv[]) {
    std::string nnClustersFile, configFile, outputFile, inputFileOriginalSpace, inputFileNewSpace;
    bool complete = false;

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


    int K = 10, L = 3, k_LSH = 4; 

    std::ifstream config_ifs(configFile);

    std::string var;
    int value;
    
    // Read configuration file
    while(config_ifs >> var >> value) {
        if(var == "number_of_clusters:") {
            K = value;
        }
        else if(var == "number_of_vector_hash_tables:") {
            L = value;
        }
        else if(var == "number_of_vector_hash_functions:") {
            k_LSH = value;
        }
        else {
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
        //Dataset<Pixel16Bit> *datasetNewSpace = new Dataset<Pixel16Bit>(inputFileNewSpace);

        if (datasetOriginalSpace->isValid()) {
            // Get images from dataset
            std::vector<Image<Pixel8Bit>*> imagesOriginalSpace = datasetOriginalSpace->getImages();
            //std::vector<Image<Pixel16Bit>*> imagesNewSpace = datasetNewSpace->getImages();

            // Initialize uniform random distribution number generator
            std::default_random_engine generator;
            std::uniform_int_distribution<int> uniform_distribution(1,imagesOriginalSpace.size());

            // k-Means++ initialization:
            // Choose a centroid uniformly at random (indexing ranges from 1 to n)
            std::unordered_set<int> centroids;
            centroids.insert(uniform_distribution(generator));
            for (int t = 1; t < K; t++) {
                // For all non-centroids i, let D(i) = min distance to some centroid, among t chosen centroids and calculate P(r) = sum{D(i), 0 <= i <= r}
                double *P = new double[imagesOriginalSpace.size() - t + 1];
                int *non_cendroid_index = new int[imagesOriginalSpace.size() - t + 1];
                P[0] = 0;
                // Calculate max{D(i)} for all non-centroids i
                unsigned long long maxDi = 0;
                for (unsigned int i = 1,j = 0; j < imagesOriginalSpace.size(); j++) {
                    // Check if jth point is not a centroid and if so, keep it's index , calculate D(i) and use it to calculate P(i) using prefix sum technique. Otherwise, continue to next point.
                    if (centroids.find(j+1) == centroids.end()) {
                        // j is not a centroid
                        // Compute D(i)
                        double D = 1.0/0.0;
                        for (auto c : centroids) {
                            double dist = imagesOriginalSpace[c-1]->distance(imagesOriginalSpace[i-1],1);
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
                for (unsigned int i = 1,j = 0; j < imagesOriginalSpace.size(); j++) {
                    // Check if jth point is not a centroid and if so, keep it's index , calculate D(i) and use it to calculate P(i) using prefix sum technique. Otherwise, continue to next point.
                    if (centroids.find(j+1) == centroids.end()) {
                        // j is not a centroid
                        // Compute D(i)
                        double D = 1.0/0.0;
                        for (auto c : centroids) {
                            double dist = imagesOriginalSpace[c-1]->distance(imagesOriginalSpace[i-1],1);
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
                std::uniform_real_distribution<float> floatDistribution(0,P[imagesOriginalSpace.size() - t]);

                // Pick a uniformly distributed float x ∈ [0,P(n−t)] and return r ∈ {1,2,...,n−t} : P(r−1) < x ≤ P(r), where P(0) = 0.
                float x = floatDistribution(generator);
                int left = 1,right = imagesOriginalSpace.size() - t,r = 0;
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
            std::vector<Cluster<Pixel8Bit>*> clusters;
            unsigned int cid = 0;
            for (auto c : centroids) {
                clusters.push_back(new Cluster<Pixel8Bit>(*imagesOriginalSpace[c-1],cid++));
            }

            unsigned int assignments;
            std::unordered_map<int, Cluster<Pixel8Bit>*> clusterHistory;

            // Clustering time!!!
            clock_t begin_clustering_time = clock();
            do {
                for (unsigned int i = 0; i < clusters.size(); i++) {
                    clusters[i]->clear();
                }
                assignments = 0;
                // Assignment step
                // Lloyd's algorithn
                for (unsigned int i = 0; i < imagesOriginalSpace.size(); i++) {
                    // Find closest cluster for the current(ith) image
                    double minDist = 1.0/0.0;
                    Cluster<Pixel8Bit> *minCluster = NULL;
                    for (unsigned j = 0; j < clusters.size(); j++) {
                        double dist = imagesOriginalSpace[i]->distance(clusters[j]->getCentroid(),1);
                        if (dist < minDist) {
                            minDist = dist;
                            minCluster = clusters[j];
                        }
                    }
                    // Insert the ith image to it's closest cluster
                    minCluster->addPoint(imagesOriginalSpace[i]);
                    if (clusterHistory.find(imagesOriginalSpace[i]->getId()) == clusterHistory.end() || clusterHistory[imagesOriginalSpace[i]->getId()]->getId() != minCluster->getId()) {
                        assignments++;
                    }
                    clusterHistory[imagesOriginalSpace[i]->getId()] = minCluster;
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
                for (int j = 0; j < datasetOriginalSpace->getImageDimension() - 1; j++) {
                    outputStream << (int)clusters[i]->getCentroid()->getPixel(j) << ", ";
                }
                outputStream << (int)clusters[i]->getCentroid()->getPixel(datasetOriginalSpace->getImageDimension()-1) << "]}\n";
            }

            // Print clustering time
            outputStream << "clustering_time: " << clustering_time << std::endl;

            // Calculate Silhouette for all images
            double averageSilhouette = 0.0;
            std::vector<double> s;
            for (unsigned int i = 0; i < imagesOriginalSpace.size(); i++) {
                // Calculate distance of ith image to all the clusters
                Cluster<Pixel8Bit> *neighbourCluster = NULL, *closestCluster = NULL;
                double distToClosestCentroid = 1.0/0.0, distToSecondClosest = 1.0/0.0;
                
                for (unsigned int j = 0; j < clusters.size(); j++) {
                    double dist = imagesOriginalSpace[i]->distance(clusters[j]->getCentroid(), 1);

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
                double ai = clusterHistory[imagesOriginalSpace[i]->getId()]->avgDistance(imagesOriginalSpace[i]);
                // Calculate average distance of ith image to images in the next best(neighbor) cluster
                double bi = neighbourCluster->avgDistance(imagesOriginalSpace[i]);
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
                std::vector<Image<Pixel8Bit>*> clusterPoints = clusters[i]->getPoints();
                for (unsigned int j = 0; j < clusterPoints.size(); j++) {
                    avgS += s[clusterPoints[j]->getId()];
                }
                outputStream << avgS/clusterPoints.size() << ",";
            }
            
            // Print average Silhouette for all points in dataset
            averageSilhouette /= imagesOriginalSpace.size();
            outputStream << " " << averageSilhouette << "]\n";

            // Optionally (with command line parameter –complete) print image numbers in each cluster
            if (complete) {
                for (unsigned int i = 0; i < clusters.size(); i++) {
                    outputStream << "CLUSTER-" << i+1 << " {[";
                    for (int j = 0; j < datasetOriginalSpace->getImageDimension() - 1; j++) {
                        outputStream << (int)clusters[i]->getCentroid()->getPixel(j) << ", ";
                    }
                    outputStream << (int)clusters[i]->getCentroid()->getPixel(datasetOriginalSpace->getImageDimension()-1) << "]";
                    std::vector<Image<Pixel8Bit>*> clusterImages = clusters[i]->getPoints();
                    for (unsigned int j = 0; j < clusterImages.size(); j++) {
                        outputStream << ", " << clusterImages[j]->getId();
                    }
                    outputStream << "}\n";
                }
            }

            for (unsigned int i = 0;i < clusters.size();i++) {
                delete clusters[i];
            }
        }
        
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
        }

    } while(repeat);

    outputStream.close();

    return 0;
}

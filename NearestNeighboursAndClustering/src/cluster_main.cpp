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
#include "../headers/lsh.h"

void usage() {
    std::cout << "Usage:./cluster  –i  <input  file>  –c  <configuration  file>  -o  <output  file>  -complete  <optional> -m <method: Classic OR LSH or Hypercube>\n";
}

double minDistBetweenClusterCentroids(std::vector<Cluster*> &clusters) {
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
    std::string inputFile, configFile, outputFile, method;
    bool complete = false;

    // Read command line arguments
    if (argc == 9 || argc == 10) {
        for(int i = 1; i < argc; i+=2) {
            std::string arg(argv[i]);

            if(!arg.compare("-i")) {
                inputFile = argv[i+1];
            }
            else if(!arg.compare("-c")) {
				configFile = argv[i+1];
			}
			else if(!arg.compare("-o")) {
				outputFile = argv[i+1];
			}
			else if(!arg.compare("-complete")) {
				complete = true;
                i--;
			}
			else if(!arg.compare("-m")) {
				method = argv[i+1];
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


    int K = 10, L = 3, k_LSH = 4, M = 10, k_hypercube = 3, probes = 2; 

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
        else if(var == "max_number_M_hypercube:") {
            M = value;
        }
        else if(var == "number_of_hypercube_dimensions:") {
            k_hypercube = value;
        }
        else if(var == "number_of_probes:") {
            probes = value;
        }
        else {
            std::cout << "Invalid conguration file" << std::endl;
            return 0;
        }
    }

    config_ifs.close();
    std::ofstream outputStream(outputFile);

    bool repeat;
    do {
        // Read Dataset
        Dataset<Image,Pixel8Bit> *dataset = new Dataset<Image,Pixel8Bit>(inputFile);

        if (dataset->isValid()) {
            // Get images from dataset
            std::vector<Image*> images = dataset->getImages();

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
            std::vector<Cluster*> clusters;
            unsigned int cid = 0;
            for (auto c : centroids) {
                clusters.push_back(new Cluster(*images[c-1],cid++));
            }

            unsigned int assignments;
            std::unordered_map<int, Cluster*> clusterHistory;

            int w = 0;
            LSH *lsh = NULL;
            Hypercube *hypercube = NULL;
            std::unordered_map<int,Image*> pointsMap;
            if (method == "LSH" || method == "Hypercube") {
                w = dataset->avg_NN_distance() * 6;
                // Initialize LSH or Hypercube interface
                if (method == "LSH") {
                    lsh = new LSH(k_LSH,w,L,dataset);
                }
                else {
                    hypercube = new Hypercube(dataset, k_hypercube, w);
                }
                // Create an unordered map with all the points to help in the reverse assignment step
                for (unsigned int i = 0; i < images.size(); i++) {
                    pointsMap[images[i]->getId()] = images[i];
                }
            }

            // Clustering time!!!
            clock_t begin_clustering_time = clock();
            do {
                for (unsigned int i = 0; i < clusters.size(); i++) {
                    clusters[i]->clear();
                }
                assignments = 0;
                // Assignment step
                if (method == "Classic") {
                    // Lloyd's algorithn
                    for (unsigned int i = 0; i < images.size(); i++) {
                        // Find closest cluster for the current(ith) image
                        double minDist = 1.0/0.0;
                        Cluster *minCluster = NULL;
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
                } else if (method == "LSH") {
                    // LSH Reverse Assignment
                    std::unordered_map<int,Image*> tmpPointsMap = pointsMap;
                    std::unordered_map<int,Cluster*> bestCluster;
                    double R = ceil(minDistBetweenClusterCentroids(clusters)/2.0);
                    unsigned int curPoints = 0,prevPoints;
                    do {
                        prevPoints = curPoints;
                        curPoints = 0;
                        // Range search on all cluster centroids and find the best cluster for all range-searched points
                        for (unsigned int i = 0; i < clusters.size(); i++) {
                            std::vector<Image*> pointsInRange = lsh->rangeSearch(clusters[i]->getCentroid(),R);
                            curPoints += pointsInRange.size();
                            for (unsigned int j = 0; j < pointsInRange.size(); j++) {
                                // Check if current in-range point was not yet assigned to a cluster
                                if (tmpPointsMap.find(pointsInRange[j]->getId()) != tmpPointsMap.end()) {
                                    // If so, assign it to the corresponding cluster
                                    bestCluster[pointsInRange[j]->getId()] = clusters[i];
                                    tmpPointsMap.erase(pointsInRange[j]->getId());
                                } else if (pointsInRange[j]->distance(clusters[i]->getCentroid(),1) < pointsInRange[j]->distance(bestCluster[pointsInRange[j]->getId()]->getCentroid(),1)) {
                                    bestCluster[pointsInRange[j]->getId()] = clusters[i];
                                }
                            }
                        }
                        R *= 2.0;
                    } while (curPoints - prevPoints > 0);
                    // Assign all range-searched points to their best cluster
                    for (auto it : bestCluster) {
                        it.second->addPoint(pointsMap[it.first]);
                        if (clusterHistory.find(it.first) == clusterHistory.end() || clusterHistory[it.first]->getId() != it.second->getId()) {
                            assignments++;
                        }
                        clusterHistory[it.first] = it.second;
                    }
                    // Assign rest points using Lloyd's method
                    for(auto it : tmpPointsMap) {
                        double distToClosestCentroid = 1.0/0.0;
                        Cluster *closestCluster = NULL;

                        for (unsigned int i = 0; i < clusters.size(); i++) {
                            double dist = it.second->distance(clusters[i]->getCentroid(),1);

                            if (dist < distToClosestCentroid) {
                                distToClosestCentroid = dist;
                                closestCluster = clusters[i];
                            }
                        }
                        // Insert the current image to it's closest cluster
                        closestCluster->addPoint(it.second);
                        if (clusterHistory.find(it.first) == clusterHistory.end() || clusterHistory[it.first]->getId() != closestCluster->getId()) {
                            assignments++;
                        }

                        clusterHistory[it.first] = closestCluster;
                    }
                } else if (method == "Hypercube") {
                    // Hypercube Reverse Assignment
                    std::unordered_map<int,Image*> tmpPointsMap = pointsMap;
                    std::unordered_map<int,Cluster*> bestCluster;
                    double R = ceil(minDistBetweenClusterCentroids(clusters)/2.0);
                    unsigned int curPoints = 0,prevPoints;
                    do {
                        prevPoints = curPoints;
                        curPoints = 0;
                        // Range search on all cluster centroids and assign the returned in-range points
                        for (unsigned int i = 0; i < clusters.size(); i++) {
                            std::list<Image*> pointsInRange = hypercube->rangeSearch(clusters[i]->getCentroid(),M,probes,R);
                            curPoints += pointsInRange.size();
                            for (std::list<Image*>::iterator it = pointsInRange.begin(); it != pointsInRange.end(); it++) {
                                // Check if current in-range point was not yet assigned to a cluster
                                if (tmpPointsMap.find((*it)->getId()) != tmpPointsMap.end()) {
                                    // If so, assign it to the corresponding cluster
                                    bestCluster[(*it)->getId()] = clusters[i];
                                    tmpPointsMap.erase((*it)->getId());
                                } else if ((*it)->distance(clusters[i]->getCentroid(),1) < (*it)->distance(bestCluster[(*it)->getId()]->getCentroid(),1)) {
                                    bestCluster[(*it)->getId()] = clusters[i];
                                }
                            }
                        }

                        R *= 2.0;
                    } while (curPoints - prevPoints > 0);
                    // Assign all range-searched points to their best cluster
                    for (auto it : bestCluster) {
                        it.second->addPoint(pointsMap[it.first]);
                        if (clusterHistory.find(it.first) == clusterHistory.end() || clusterHistory[it.first]->getId() != it.second->getId()) {
                            assignments++;
                        }
                        clusterHistory[it.first] = it.second;
                    }
                    // Assign rest points using Lloyd's method
                    for(auto it : tmpPointsMap) {
                        double distToClosestCentroid = 1.0/0.0;
                        Cluster *closestCluster = NULL;
                        
                        for (unsigned int i = 0; i < clusters.size(); i++) {
                            double dist = it.second->distance(clusters[i]->getCentroid(),1);
                            if (dist < distToClosestCentroid) {
                                distToClosestCentroid = dist;
                                closestCluster = clusters[i];
                            }
                        }
                        // Insert the current image to it's closest cluster
                        closestCluster->addPoint(it.second);
                        if (clusterHistory.find(it.first) == clusterHistory.end() || clusterHistory[it.first]->getId() != closestCluster->getId()) {
                            assignments++;
                        }
                        clusterHistory[it.first] = closestCluster;
                    }
                }

                // Update all cluster centroids
                if (assignments > 0) {
                    for (unsigned int i = 0; i < clusters.size(); i++) {
                        clusters[i]->updateCentroid();
                    }
                }
            } while (assignments > 100);
            double clustering_time = double(clock() - begin_clustering_time) / CLOCKS_PER_SEC;
            
            // Print used method
            outputStream << "Algorithm: ";
            if (method == "Classic") {
                outputStream << "Lloyds";
            } else if (method == "LSH") {
                outputStream << "Range Search LSH";
            } else if (method == "Hypercube") {
                outputStream << "Range Search Hypercube";
            }
            outputStream << std::endl;

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
            double averageSilhouette = 0.0;
            std::vector<double> s;
            for (unsigned int i = 0; i < images.size(); i++) {
                // Calculate distance of ith image to all the clusters
                Cluster *neighbourCluster = NULL, *closestCluster = NULL;
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
                std::vector<Image*> clusterPoints = clusters[i]->getPoints();
                for (unsigned int j = 0; j < clusterPoints.size(); j++) {
                    avgS += s[clusterPoints[j]->getId()];
                }
                outputStream << avgS/clusterPoints.size() << ",";
            }
            
            // Print average Silhouette for all points in dataset
            averageSilhouette /= images.size();
            outputStream << " " << averageSilhouette << "]\n";

            // Optionally (with command line parameter –complete) print image numbers in each cluster
            if (complete) {
                for (unsigned int i = 0; i < clusters.size(); i++) {
                    outputStream << "CLUSTER-" << i+1 << " {[";
                    for (int j = 0; j < dataset->getImageDimension() - 1; j++) {
                        outputStream << (int)clusters[i]->getCentroid()->getPixel(j) << ", ";
                    }
                    outputStream << (int)clusters[i]->getCentroid()->getPixel(dataset->getImageDimension()-1) << "]";
                    std::vector<Image*> clusterImages = clusters[i]->getPoints();
                    for (unsigned int j = 0; j < clusterImages.size(); j++) {
                        outputStream << ", " << clusterImages[j]->getId();
                    }
                    outputStream << "}\n";
                }
            }

            for (unsigned int i = 0;i < clusters.size();i++) {
                delete clusters[i];
            }
            
            if (method == "LSH") {
                delete lsh;
            }
            if (method == "Hypercube") {
                delete hypercube;
            }
        }
        
        delete dataset;

        std::string promptAnswer;
        std::cout << "Do you want to enter another query (Y/N)?  ";

        do {
            std::cin >> promptAnswer;
        } while(promptAnswer != "Y" && promptAnswer != "N" && std::cout << "Invalid answer" << std::endl);

        repeat = (promptAnswer == "Y");

        if(repeat) {
            std::cout << "Input File: ";
            std::cin >> inputFile;
        }

    } while(repeat);

    outputStream.close();

    return 0;
}

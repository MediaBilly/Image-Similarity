#include <iostream>
#include <string>
#include <cmath>
#include "../headers/dataset.h"
#include "../headers/image.h"
#include "../headers/utilities.h"
#include "ortools/linear_solver/linear_solver.h"

double eucledian_distance(int x1,int y1,int x2,int y2) {
    return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

double calculate_emd(Image<Pixel8Bit>* image_1, Image<Pixel8Bit>* image_2,int clusterDimension) {
    using namespace operations_research;
    // Generate the clusters for the 2 images
    std::vector<Image<Pixel8Bit>*> clusters_1 = image_1->clusters(clusterDimension);
    std::vector<Image<Pixel8Bit>*> clusters_2 = image_2->clusters(clusterDimension);

    // Create a linear solver
    MPSolver* linearSolver = MPSolver::CreateSolver("GLOP");

    double infinity = linearSolver->infinity();

    // Define the flow variables (fij) and calculate the distances between their centroids
    MPVariable *flow[clusters_1.size()][clusters_2.size()];
    double d[clusters_1.size()][clusters_2.size()];
    for (unsigned int i = 0; i < clusters_1.size(); i++) {
        std::tuple<int,int> centroid1 = clusters_1[i]->findCentroid();
        int x1 = (clusters_1[i]->getId() % (clusters_1[i]->getWidth() / clusters_1[i]->getWidth())) * clusters_1[i]->getWidth() + std::get<0>(centroid1);
        int y1 = (clusters_1[i]->getId() / (clusters_1[i]->getWidth() / clusters_1[i]->getWidth())) * clusters_1[i]->getHeight() + std::get<1>(centroid1);
        
        for (unsigned int j = 0; j < clusters_2.size(); j++) {
            flow[i][j] = linearSolver->MakeNumVar(0, infinity, "flow_" + std::to_string(i) + "_" + std::to_string(j));
            // Calculate the distances between the 2 image centroids
            std::tuple<int,int> centroid2 = clusters_2[j]->findCentroid();
            int x2 = (clusters_2[j]->getId() % (clusters_2[j]->getWidth() / clusters_2[j]->getWidth())) * clusters_2[j]->getWidth() + std::get<0>(centroid2);
            int y2 = (clusters_2[j]->getId() / (clusters_2[j]->getWidth() / clusters_2[j]->getWidth())) * clusters_2[j]->getHeight() + std::get<1>(centroid2);

            d[i][j] = eucledian_distance(x1,y1,x2,y2);
        }
    }
    
    // Define the constraints
    MPConstraint *c0[clusters_1.size()];
    unsigned int image_1_totalValue = image_1->totalValue();
    for (unsigned int i = 0;i < clusters_1.size();i++) {
        double w = clusters_1[i]->totalValue()/image_1_totalValue;
        c0[i] = linearSolver->MakeRowConstraint(w, w, "c0_" + std::to_string(i));
        // Set the coefficients
        for (int j = 0;j < clusters_2.size();j++) {
            c0[i]->SetCoefficient(flow[i][j], 1);
        }
    }

    MPConstraint *c1[clusters_2.size()];
    unsigned int image_2_totalValue = image_2->totalValue();
    for (unsigned int j = 0;j < clusters_2.size();j++) {
        double w = clusters_2[j]->totalValue()/image_2_totalValue;
        c0[j] = linearSolver->MakeRowConstraint(w, w, "c1_" + std::to_string(j));
        // Set the coefficients
        for (int i = 0;i < clusters_1.size();i++) {
            c0[j]->SetCoefficient(flow[i][j], 1);
        }
    }

    // Define minimization objective
    MPObjective* objective = linearSolver->MutableObjective();
    for (unsigned int i = 0; i < clusters_1.size(); i++) {
        for (unsigned int j = 0; j < clusters_2.size(); j++) {
            objective->SetCoefficient(flow[i][j], d[i][j]);
        }
    }
    objective->SetMinimization();

    // Solve
    MPSolver::ResultStatus result_status = linearSolver->Solve();

    double result = objective->Value();

    // Destroy all the clusters
    for (std::vector<Image<Pixel8Bit>*>::iterator it = clusters_1.begin();it != clusters_1.end();it++) {
        delete *it;
    }
    for (std::vector<Image<Pixel8Bit>*>::iterator it = clusters_2.begin();it != clusters_2.end();it++) {
        delete *it;
    }

    return result;
}


int main(int argc, char const *argv[]) {
    // Load training dataset
    std::string trainDatasetPath = "../datasets/training/images.dat";
    Dataset<Pixel8Bit> *trainDataset = new Dataset<Pixel8Bit>(trainDatasetPath);
    std::vector<Image<Pixel8Bit>*> trainImages = trainDataset->getImages();

    // Load query dataset
    std::string queryDatasetPath = "../datasets/training/images.dat";
    Dataset<Pixel8Bit> *queryDataset = new Dataset<Pixel8Bit>(queryDatasetPath);
    std::vector<Image<Pixel8Bit>*> queryImages = queryDataset->getImages();
    
    int cluster_dimension = 4;
    for (std::vector<Image<Pixel8Bit>*>::iterator qit = queryImages.begin();qit != queryImages.end(); qit++) {
        std::cout << "Query: " << (*qit)->getId() << std::endl;
        for (std::vector<Image<Pixel8Bit>*>::iterator tit = trainImages.begin();tit != trainImages.end(); tit++) {
            std::cout << "EMD to Image " << (*tit)->getId() << ": " << calculate_emd(*qit,*tit,cluster_dimension) << std::endl;
        }
        std::cout << std::endl;
    }
    

    // double clusterWeight[trainDataset->getImageDimension()/(cluster_dimension * cluster_dimension)];

    // for (std::vector<Image<Pixel8Bit>*>::iterator it = images.begin();it != images.end();it++) {
    //     std::cout << "Image " << it - images.begin() << std::endl;

    //     std::vector<Image<Pixel8Bit>*> clusters = (*it)->clusters(cluster_dimension);
    //     double imageTotalValue = (double)(*it)->totalValue();

    //     for (std::vector<Image<Pixel8Bit>*>::iterator itc = clusters.begin();itc != clusters.end(); itc++) {
    //         // Get coordinates of cluster centroid in image
    //         std::tuple<int,int> centroid = (*itc)->findCentroid();
    //         int x_axis = ((*itc)->getId() % ((*it)->getWidth() / (*itc)->getWidth())) * (*itc)->getWidth() + std::get<0>(centroid);
    //         int y_axis = ((*itc)->getId() / ((*it)->getWidth() / (*itc)->getWidth())) * (*itc)->getHeight() + std::get<1>(centroid);

    //         clusterWeight[(*itc)->getId()] = (*itc)->totalValue() / imageTotalValue;

    //         std::cout << "  Centroid " << (*itc)->getId() << ": (" << x_axis << "," << y_axis << ") = " << clusterWeight[(*itc)->getId()] << std::endl;
    //         delete *itc;
    //     }
    // }
    
    delete queryDataset;
    delete trainDataset;
    return 0;
}

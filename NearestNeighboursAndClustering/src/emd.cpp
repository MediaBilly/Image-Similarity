#include <iostream>
#include <string>
#include "../headers/dataset.h"
#include "../headers/image.h"
#include "../headers/utilities.h"

int main(int argc, char const *argv[]) {
    std::string datasetPath = "../datasets/training/images.dat";
    Dataset<Pixel8Bit> *dataset = new Dataset<Pixel8Bit>(datasetPath);
    std::vector<Image<Pixel8Bit>*> images = dataset->getImages();
    
    unsigned int tv,clusterSum;
    for (std::vector<Image<Pixel8Bit>*>::iterator it = images.begin();it != images.end();it++) {
        tv = (*it)->totalValue();
        clusterSum = 0;
        std::vector<Image<Pixel8Bit>*> clusters = (*it)->clusters(4);
        for (std::vector<Image<Pixel8Bit>*>::iterator itc = clusters.begin();itc != clusters.end(); itc++) {
            clusterSum += (*itc)->totalValue();
            delete *itc;
        }
        std::cout << tv << " " << clusterSum << std::endl;
    }
    
    delete dataset;
    return 0;
}

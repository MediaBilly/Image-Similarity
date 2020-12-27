#include <iostream>
#include <string>
#include "../headers/dataset.h"
#include "../headers/image.h"
#include "../headers/utilities.h"

int main(int argc, char const *argv[]) {
    std::string datasetPath = "../datasets/training/images.dat";
    Dataset<Pixel8Bit> *dataset = new Dataset<Pixel8Bit>(datasetPath);
    std::vector<Image<Pixel8Bit>*> images = dataset->getImages();
    
    unsigned int cnt = 0;
    for (std::vector<Image<Pixel8Bit>*>::iterator it = images.begin();it != images.end();it++) {
        std::vector<Image<Pixel8Bit>*> clusters = (*it)->clusters(4);
        std::cout << "Image " << cnt++ << ")" << std::endl;
        unsigned int cnt2 = 0;
        std::tuple<int,int> centroid;
        for (std::vector<Image<Pixel8Bit>*>::iterator itc = clusters.begin();itc != clusters.end(); itc++) {
            centroid = (*itc)->findCentroid();
            std::cout << "Centroid " << cnt2++ << ": (" << std::get<0>(centroid) << "," << std::get<1>(centroid) << ")" << std::endl;
            delete *itc;
        }
        std::cout << std::endl;
    }
    
    delete dataset;
    return 0;
}

#include "../headers/bruteforce_search.h"
#include <algorithm>

template<typename ImageType>
Bruteforce_Search<ImageType>::Bruteforce_Search(std::vector<ImageType*> images) {
    this->images = images;
}

template<typename ImageType>
std::vector<std::pair<double,int>> Bruteforce_Search<ImageType>::exactNN(ImageType *q, int N) {
    // Calculate distance of q to all the points in the dataset
    std::vector<std::pair<double,int>> neighbors;
    
    for (unsigned i = 0; i < this->images.size(); i++) {
        std::pair<double,int> temp(q->distance(this->images[i], 1), this->images[i]->getId());
        neighbors.push_back(temp);
    }
    
    std::sort(neighbors.begin(), neighbors.end());
    neighbors.resize(N);
    return neighbors;
}

template<typename ImageType>
Bruteforce_Search<ImageType>::~Bruteforce_Search() {

}

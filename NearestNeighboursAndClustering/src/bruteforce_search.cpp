#include "../headers/bruteforce_search.h"
#include <algorithm>

template<typename PixelType>
Bruteforce_Search<PixelType>::Bruteforce_Search(std::vector<Image<PixelType>*> images) {
    this->images = images;
}

template<typename PixelType>
std::vector<std::pair<double,int>> Bruteforce_Search<PixelType>::exactNN(Image<PixelType> *q, int N) {
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

template<typename PixelType>
Bruteforce_Search<PixelType>::~Bruteforce_Search() {

}

template class Bruteforce_Search<Pixel8Bit>;
template class Bruteforce_Search<Pixel16Bit>;

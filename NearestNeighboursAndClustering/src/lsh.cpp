#include "../headers/lsh.h"
#include <algorithm>
#include <unordered_set> 
#include <iostream>

LSH::LSH(int k,int w,int L, Dataset<Pixel8Bit> *imageDataset) {
    this->k = k;
    this->w = w;
    this->L = L;
    this->images = imageDataset->getImages();

    // Create hash tables
    this->hashTables = new Hash_Table*[L];
    for (int i = 0; i < L; i++) {
        this->hashTables[i] = new Hash_Table(k, w, imageDataset->getImageDimension(),this->images.size()/16);
    }

    // Insert all images there
    for (unsigned int i = 0; i < this->images.size(); i++) {
        for (int j = 0;j < L;j++) {
            this->hashTables[j]->insert(this->images[i]);
        }
    }
}


std::vector<std::pair<double, int>> LSH::approximate_kNN(Image<Pixel8Bit> *q, unsigned int N) {
    std::vector<std::pair<double,int>> neighbors;
    std::unordered_set<int> visited;

    for (int i = 0; i < L; i++) {
        std::list<Image<Pixel8Bit>*> bucket = this->hashTables[i]->getBucketImages(q);
        for (std::list<Image<Pixel8Bit>*>::iterator p = bucket.begin(); p != bucket.end(); p++) {
            if (visited.find((*p)->getId()) == visited.end()) {
                visited.insert((*p)->getId());
                std::pair<double,int> tmp(q->distance(*p,1),(*p)->getId());
                neighbors.push_back(tmp);
            }
        }
    }
    std::sort(neighbors.begin(),neighbors.end());
    if (neighbors.size() > N) {
        neighbors.resize(N);
    }
    return neighbors;
}

std::vector<Image<Pixel8Bit>*> LSH::rangeSearch(Image<Pixel8Bit> *q, double r) {
    std::vector<Image<Pixel8Bit>*> neighbors;
    std::unordered_set<int> visited;

    for (int i = 0; i < L; i++) {
        std::list<Image<Pixel8Bit>*> bucket = this->hashTables[i]->getBucketImages(q);
        for (std::list<Image<Pixel8Bit>*>::iterator p = bucket.begin(); p != bucket.end(); p++) {
            if (visited.find((*p)->getId()) == visited.end() && q->distance(*p,1) <= r) {
                visited.insert((*p)->getId());
                neighbors.push_back(*p);
            }
        }
    }

    return neighbors;
}

LSH::~LSH() {
    // Destroy all hash tables
    for (int i = 0; i < this->L; i++) {
        delete this->hashTables[i];
    }
    delete[] this->hashTables;
}
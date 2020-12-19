#include "../headers/cluster.h"
#include <algorithm>
#include <cmath>
#include <vector>

template<typename PixelType>
Cluster<PixelType>::Cluster(Image<PixelType> &centroid,unsigned int id) {
    this->centroid = new Image(centroid);
    this->id = id;
}

template<typename PixelType>
unsigned int Cluster<PixelType>::getId() {
    return this->id;
}

template<typename PixelType>
bool Cluster<PixelType>::addPoint(Image<PixelType> *point) {
    // Check if this point is already in this cluster
    if (this->points.find(point->getId()) == this->points.end()) {
        this->points[point->getId()] = point;
        return true;
    }

    return false;
}

template<typename PixelType>
bool Cluster<PixelType>::removePoint(int id) {
    // Check if this point is in this cluster
    if (this->points.find(id) != this->points.end()) {
        this->points.erase(id);
        return true;
    }
    
    return false;
}

template<typename PixelType>
unsigned int Cluster<PixelType>::getSize() {
    return this->points.size();
}

template<typename PixelType>
struct PointsComparator {
    PointsComparator(int d) {
        this->d = d; 
    }

    bool operator () (Image<PixelType>* p1, Image<PixelType>* p2) {
        return p1->getPixel(d) < p2->getPixel(d);
    }

    int d;
};

template<typename PixelType>
void Cluster<PixelType>::updateCentroid() {
    // Update the centroid to be the median of all the cluster's points
    for (int j = 0; j < this->centroid->getSize(); j++) {
        // Sort the points in ascending order by jth dimension
        std::vector<Image<PixelType>*> points_sorted_by_d;
        for (auto it : this->points) { 
            points_sorted_by_d.push_back(it.second);
        } 
        std::sort(points_sorted_by_d.begin(), points_sorted_by_d.end(), PointsComparator(j));
        // Set the jth pixel to the jth pixel of the median point of the sorted ones
        this->centroid->setPixel(j, points_sorted_by_d[(int) ceil(this->points.size() / 2)]->getPixel(j));
    }
}

template<typename PixelType>
void Cluster<PixelType>::clear() {
    this->points.clear();
}

template<typename PixelType>
Image<PixelType>* Cluster<PixelType>::getCentroid() {
    return this->centroid;
}

template<typename PixelType>
std::vector<Image<PixelType>*> Cluster<PixelType>::getPoints() {
    std::vector<Image<PixelType>*> ret;

    for (auto it : this->points) {
        ret.push_back(it.second);
    }

    return ret;
}

template<typename PixelType>
double Cluster<PixelType>::avgDistance(Image<PixelType> *point) {
    double dist = 0.0;

    for (auto it : this->points) {
        dist += point->distance(it.second, 1);
    }

    return dist / this->points.size();
}

template<typename PixelType>
Cluster<PixelType>::~Cluster() {
    delete this->centroid;
}


template class Cluster<Pixel8Bit>;
template class Cluster<Pixel16Bit>;
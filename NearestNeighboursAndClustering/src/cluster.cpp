#include "../headers/cluster.h"
#include <algorithm>
#include <cmath>
#include <vector>

Cluster::Cluster(Image &centroid,unsigned int id) {
    this->centroid = new Image(centroid);
    this->id = id;
}

unsigned int Cluster::getId() {
    return this->id;
}


bool Cluster::addPoint(Image *point) {
    // Check if this point is already in this cluster
    if (this->points.find(point->getId()) == this->points.end()) {
        this->points[point->getId()] = point;
        return true;
    }

    return false;
}

bool Cluster::removePoint(int id) {
    // Check if this point is in this cluster
    if (this->points.find(id) != this->points.end()) {
        this->points.erase(id);
        return true;
    }
    
    return false;
}

unsigned int Cluster::getSize() {
    return this->points.size();
}


struct PointsComparator {
    PointsComparator(int d) {
        this->d = d; 
    }

    bool operator () (Image* p1, Image* p2) {
        return p1->getPixel(d) < p2->getPixel(d);
    }

    int d;
};

void Cluster::updateCentroid() {
    // Update the centroid to be the median of all the cluster's points
    for (int j = 0; j < this->centroid->getSize(); j++) {
        // Sort the points in ascending order by jth dimension
        std::vector<Image*> points_sorted_by_d;
        for (auto it : this->points) { 
            points_sorted_by_d.push_back(it.second);
        } 
        std::sort(points_sorted_by_d.begin(), points_sorted_by_d.end(), PointsComparator(j));
        // Set the jth pixel to the jth pixel of the median point of the sorted ones
        this->centroid->setPixel(j, points_sorted_by_d[(int) ceil(this->points.size() / 2)]->getPixel(j));
    }
}

void Cluster::clear() {
    this->points.clear();
}

Image* Cluster::getCentroid() {
    return this->centroid;
}

std::vector<Image*> Cluster::getPoints() {
    std::vector<Image*> ret;

    for (auto it : this->points) {
        ret.push_back(it.second);
    }

    return ret;
}

double Cluster::avgDistance(Image *point) {
    double dist = 0.0;

    for (auto it : this->points) {
        dist += point->distance(it.second, 1);
    }

    return dist / this->points.size();
}


Cluster::~Cluster() {
    delete this->centroid;
}

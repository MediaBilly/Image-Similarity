#pragma once

#include <vector>
#include <unordered_map>
#include "image.h"

class Cluster
{
    private:
        unsigned int id;
        Image *centroid;
        std::unordered_map<int,Image*> points;
    public:
        Cluster(Image &centroid,unsigned int id);
        unsigned int getId();
        bool addPoint(Image* point);
        bool removePoint(int id);
        unsigned int getSize();
        void updateCentroid();
        void clear();
        Image* getCentroid();
        std::vector<Image*> getPoints();
        double avgDistance(Image *point);
        ~Cluster();
};


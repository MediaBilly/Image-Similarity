#pragma once

#include <vector>
#include <unordered_map>
#include "image.h"

template<typename PixelType>
class Cluster
{
    private:
        unsigned int id;
        Image<PixelType> *centroid;
        std::unordered_map<int,Image<PixelType>*> points;
    public:
        Cluster(Image<PixelType> &centroid,unsigned int id);
        unsigned int getId();
        bool addPoint(Image<PixelType>* point);
        bool removePoint(int id);
        unsigned int getSize();
        void updateCentroid();
        void clear();
        Image<PixelType>* getCentroid();
        std::vector<Image<PixelType>*> getPoints();
        double avgDistance(Image<PixelType> *point);
        ~Cluster();
};


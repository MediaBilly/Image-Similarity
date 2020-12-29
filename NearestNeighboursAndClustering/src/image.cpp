#include "../headers/image.h"
#include "../headers/utilities.h"
#include <cmath>
#include <iostream>

template<typename PixelType>
Image<PixelType>::Image(int id,int width,int height) {
    this->id = id;
    this->width = width;
    this->height = height;
    this->pixels = new Pixel8Bit[width*height];
}

// Copy constructor
template<typename PixelType>
Image<PixelType>::Image(Image &img) {
    this->id = img.id;
    this->width = img.width;
    this->height = img.height;
    // Copy pixels
    this->pixels = new Pixel8Bit[width*height];
    for (int i = 0; i < width*height; i++) {
        this->pixels[i] = img.pixels[i];
    }
    
}


template<typename PixelType>
bool Image<PixelType>::setPixel(int index,PixelType pixel) {
    // Check bounds
    if (index >= this->getSize() || index < 0) {
        return false;
    }
    
    // Set pixel color
    this->pixels[index] = pixel;
    return true;
}

template<typename PixelType>
int Image<PixelType>::getId() {
    return this->id;
}

template<typename PixelType>
int Image<PixelType>::getHeight() {
    return this->height;
}

template<typename PixelType>
int Image<PixelType>::getWidth() {
    return this->width;
}

template<typename PixelType>
PixelType Image<PixelType>::getPixel(int index) {
    return (index < this->width*this->height && index >= 0) ? this->pixels[index] : -1;
}

template<typename PixelType>
int Image<PixelType>::getSize() {
    return this->width * this->height;
}

template<typename PixelType>
double Image<PixelType>::distance(Image *image, int norm) {
    // Check if both images lie on the same space
    if (this->getSize() != image->getSize()) {
        return -1.0;
    }
    
    double d = 0.0;
    for (int i = 0; i < this->getSize(); i++) {
        d += power(abs(this->getPixel(i) - image->getPixel(i)),norm);
    }
    return pow(d, 1/norm);
}

template<typename PixelType>
unsigned int Image<PixelType>::totalValue() {
    unsigned int total = 0;
    for (int i = 0;i < this->getSize();i++) {
        total += this->pixels[i];
    }
    return total;
}

template<typename PixelType>
Image<PixelType>::~Image() {
    delete[] this->pixels;
}

template<typename PixelType>
void Image<PixelType>::normalize(unsigned int normed_sum) {
    unsigned int sum = this->totalValue();

    for(int i = 0; i < this->getSize(); i++) {        
        this->pixels[i] *= normed_sum / sum;
    }
}

template<typename PixelType>
std::vector<Image<PixelType>*> Image<PixelType>::clusters(int clusterDimension) {
    std::vector<Image<PixelType>*> ret;
    
    if(this->height % clusterDimension || this->width % clusterDimension)
        return ret;

    // Create the clusters
    for (int i = 0; i < this->getSize() / (clusterDimension * clusterDimension); i++) {
        ret.push_back(new Image<PixelType>(i, clusterDimension, clusterDimension));
    }

    // Set every pixel to it's corresponding cluster
    for(int i = 0; i < this->getSize(); i++) {
        int x_axis = i % this->width;
        int y_axis = (i / this->width) % this->height;

        int x_cluster_axis = (x_axis / clusterDimension) % (this->width / clusterDimension);
        int y_cluster_axis = (y_axis / clusterDimension) % (this->height / clusterDimension);

        int currCluster = y_cluster_axis * (this->width / clusterDimension) + x_cluster_axis;

        int x_inner_axis = x_axis % clusterDimension;
        int y_inner_axis = y_axis % clusterDimension;
        int currPixel = y_inner_axis * clusterDimension + x_inner_axis;
        
        ret[currCluster]->setPixel(currPixel, this->pixels[i]);
    }

    // Return the clusters
    return ret;
}


template<typename PixelType>
std::tuple<int,int> Image<PixelType>::findCentroid() {
    int maxValue = 0;
    int weighted_sum = 0;
    
    for(int i = 0; i < this->getSize(); i++) {
        weighted_sum += this->pixels[i];

        if(this->pixels[i] > maxValue)
            maxValue = this->pixels[i];
    }
    
    int centroid_index = maxValue ? weighted_sum / this->getSize() : this->getSize() / 2;
    int x_axis = centroid_index % this->width;
    int y_axis = (centroid_index / this->width) % this->height;

    std::tuple<int,int> ret(x_axis, y_axis);
    return ret;
}

template class Image<Pixel8Bit>;
template class Image<Pixel16Bit>;
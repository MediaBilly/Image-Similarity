#include "../headers/imageReduced.h"
#include "../headers/utilities.h"
#include <cmath>
#include <iostream>

ImageReduced::ImageReduced(int id,int width,int height) {
    this->id = id;
    this->width = width;
    this->height = height;
    this->pixels = new Pixel16Bit[width*height];
}

// Copy constructor
ImageReduced::ImageReduced(ImageReduced &img) {
    this->id = img.id;
    this->width = img.width;
    this->height = img.height;
    // Copy pixels
    this->pixels = new Pixel16Bit[width*height];
    for (int i = 0; i < width*height; i++) {
        this->pixels[i] = img.pixels[i];
    }
    
}



bool ImageReduced::setPixel(int index,Pixel16Bit pixel) {
    // Check bounds
    if (index >= this->getSize() || index < 0) {
        return false;
    }
    
    // Set pixel color
    this->pixels[index] = pixel;
    return true;
}

int ImageReduced::getId() {
    return this->id;
}

Pixel16Bit ImageReduced::getPixel(int index) {
    return (index < this->width*this->height && index >= 0) ? this->pixels[index] : -1;
}

int ImageReduced::getSize() {
    return this->width * this->height;
}

double ImageReduced::distance(ImageReduced *image, int norm) {
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

ImageReduced::~ImageReduced() {
    delete[] this->pixels;
}
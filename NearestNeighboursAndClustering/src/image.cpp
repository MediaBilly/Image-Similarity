#include "../headers/image.h"
#include "../headers/utilities.h"
#include <cmath>
#include <iostream>

Image::Image(int id,int width,int height) {
    this->id = id;
    this->width = width;
    this->height = height;
    this->pixels = new Pixel8Bit[width*height];
}

// Copy constructor
Image::Image(Image &img) {
    this->id = img.id;
    this->width = img.width;
    this->height = img.height;
    // Copy pixels
    this->pixels = new Pixel8Bit[width*height];
    for (int i = 0; i < width*height; i++) {
        this->pixels[i] = img.pixels[i];
    }
    
}



bool Image::setPixel(int index,Pixel8Bit pixel) {
    // Check bounds
    if (index >= this->getSize() || index < 0) {
        return false;
    }
    
    // Set pixel color
    this->pixels[index] = pixel;
    return true;
}

int Image::getId() {
    return this->id;
}

Pixel8Bit Image::getPixel(int index) {
    return (index < this->width*this->height && index >= 0) ? this->pixels[index] : -1;
}

int Image::getSize() {
    return this->width * this->height;
}

double Image::distance(Image *image, int norm) {
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

Image::~Image() {
    delete[] this->pixels;
}
#include "../headers/dataset.h"
#include "../headers/bruteforce_search.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cerrno>

#define SWAP_INT16(x) ((((unsigned short)(x)) >> 8) | (((unsigned short)(x)) << 8))
#define SWAP_INT32(x) ((((unsigned int)(x)) >> 24) | ((((unsigned int)(x)) & 0x00FF0000) >> 8) | ((((unsigned int)(x)) & 0x0000FF00) << 8) | (((unsigned int)(x)) << 24))
#define W_SAMPLE_SIZE 50

template<typename ImageType, typename PixelType>
Dataset<ImageType,PixelType>::Dataset(std::string inputPath, int bytes_per_pixel)
{   
    // Open dataset binary file
    std::ifstream input(inputPath, std::ios::out | std::ios::binary);
    if(!input) {
        std::cout << "Cannot open file!:" << strerror(errno) << std::endl;
        this->valid = false;
        return;
    }
    // Read header
    input.read((char*)&this->head,sizeof(header));
    
    // Swap endianess
    this->head.magic_num = SWAP_INT32(this->head.magic_num);
    this->head.num_of_images = SWAP_INT32(this->head.num_of_images);
    this->head.num_of_rows = SWAP_INT32(this->head.num_of_rows);
    this->head.num_of_columns = SWAP_INT32(this->head.num_of_columns);
    this->bytes_per_pixel = bytes_per_pixel;

    // TODO:Read images
    for (unsigned int i = 0; i < this->head.num_of_images; i++) {
        // Read pixels for every image
        this->images.push_back(new Image(i, this->head.num_of_columns, this->head.num_of_rows));
        for (int p = 0; p < this->getImageDimension(); p++) {
            PixelType pix;
            input.read((char*)&pix, sizeof(PixelType));
            this->images.at(i)->setPixel(p,pix);
        }
    }
    // Close dataset binary file
    input.close();

    this->valid = true;
}

template<typename ImageType, typename PixelType>
bool Dataset<ImageType,PixelType>::isValid() {
    return this->valid;
}

template<typename ImageType, typename PixelType>
int Dataset<ImageType,PixelType>::getImageDimension() {
    return this->head.num_of_rows * this->head.num_of_columns;
}

template<typename ImageType, typename PixelType>
int Dataset<ImageType,PixelType>::avg_NN_distance() {
    int step = images.size() / W_SAMPLE_SIZE;
    double dist_sum = 0.0;
    Bruteforce_Search bf(images);

    for(unsigned int i = 0; i < images.size(); i += step) {
        dist_sum += bf.exactNN(images[i], 2)[1].first;
    }

    return dist_sum / W_SAMPLE_SIZE;
}

template<typename ImageType, typename PixelType>
std::vector<ImageType*> Dataset<ImageType,PixelType>::getImages(){
    return this->images;
}

template<typename ImageType, typename PixelType>
Dataset<ImageType,PixelType>::~Dataset() {
    for (std::vector<ImageType*>::iterator it = this->images.begin();it < this->images.end();it++) {
        delete *it;
    }
}
#include "../headers/dataset.h"
#include "../headers/bruteforce_search.h"
#include "../headers/utilities.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cerrno>

#define W_SAMPLE_SIZE 50

template<typename PixelType>
Dataset<PixelType>::Dataset(std::string inputPath)
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

    // TODO:Read images
    for (unsigned int i = 0; i < this->head.num_of_images; i++) {
        // Read pixels for every image
        this->images.push_back(new Image<PixelType>(i, this->head.num_of_columns, this->head.num_of_rows));
        for (int p = 0; p < this->getImageDimension(); p++) {
            PixelType pix;
            input.read((char*)&pix, sizeof(PixelType));
            if (sizeof(PixelType) == sizeof(Pixel16Bit)) {
                pix = SWAP_INT16(pix);
            }
            this->images.at(i)->setPixel(p,pix);
        }
    }
    // Close dataset binary file
    input.close();

    this->valid = true;
}

template<typename PixelType>
bool Dataset<PixelType>::isValid() {
    return this->valid;
}

template<typename PixelType>
int Dataset<PixelType>::getImageDimension() {
    return this->head.num_of_rows * this->head.num_of_columns;
}

template<typename PixelType>
int Dataset<PixelType>::getImageWidth() {
    return this->head.num_of_columns;
}

template<typename PixelType>
int Dataset<PixelType>::getImageHeight() {
    return this->head.num_of_rows;
}

template<typename PixelType>
int Dataset<PixelType>::avg_NN_distance() {
    int step = images.size() / W_SAMPLE_SIZE;
    double dist_sum = 0.0;
    Bruteforce_Search<PixelType> *bf = new Bruteforce_Search<PixelType>(images);

    for(unsigned int i = 0; i < images.size(); i += step) {
        dist_sum += bf->exactNN(images[i], 2)[1].first;
    }

    delete bf;
    return dist_sum / W_SAMPLE_SIZE;
}

template<typename PixelType>
std::vector<Image<PixelType>*> Dataset<PixelType>::getImages(){
    return this->images;
}

template<typename PixelType>
Dataset<PixelType>::~Dataset() {
    for (typename std::vector<Image<PixelType>*>::iterator it = this->images.begin();it < this->images.end();it++) {
        delete *it;
    }
}

template class Dataset<Pixel8Bit>;
template class Dataset<Pixel16Bit>;
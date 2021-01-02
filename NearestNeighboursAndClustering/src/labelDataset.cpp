#include "../headers/labelDataset.h"
#include "../headers/utilities.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstring>
#include <cerrno>

LabelDataset::LabelDataset(std::string inputPath) {
    // Open labels binary file
    std::ifstream input(inputPath, std::ios::out | std::ios::binary);
    if(!input) {
        std::cout << "Cannot open file!:" << strerror(errno) << std::endl;
        this->valid = false;
        return;
    }

    // Read magic number
    input.read((char*)&this->magic_num,sizeof(this->magic_num));
    this->magic_num = SWAP_INT32(this->magic_num);

    // Read number of items
    input.read((char*)&this->num_of_items,sizeof(this->num_of_items));
    this->num_of_items = SWAP_INT32(this->num_of_items);

    // Read all labels
    uint8_t label;
    for (unsigned int i = 0;i < this->num_of_items;i++) {
        input.read((char*)&label,sizeof(label));
        this->labels.push_back(label);
    }

    // Close dataset binary file
    input.close();

    this->valid = true;
}

bool LabelDataset::isValid() {
    return this->valid;
}

uint8_t LabelDataset::getLabel(int labelIndex) {
    if (labelIndex < 0 || labelIndex >= this->labels.size()) {
        return -1;
    }
    return this->labels[labelIndex];
}

LabelDataset::~LabelDataset() {
    
}
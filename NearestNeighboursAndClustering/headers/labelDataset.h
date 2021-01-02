#pragma once

#include <string>
#include <vector>

class LabelDataset
{
    private:
        unsigned int magic_num;
        unsigned int num_of_items;
        bool valid;
        std::vector<uint8_t> labels;
    public:
        LabelDataset(std::string inputPath);
        bool isValid();
        uint8_t getLabel(int imgIndex);
        ~LabelDataset();
};
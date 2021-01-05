#pragma once

#include <string>
#include <vector>
#include "image.h"

template<typename PixelType>
class Dataset
{
    private:
        struct header
        {
            unsigned int magic_num;
            unsigned int num_of_images;
            unsigned int num_of_rows;
            unsigned int num_of_columns;
        };
        bool valid;
        header head;
        std::vector<Image<PixelType>*> images;

    public:
        Dataset(std::string inputPath);
        bool isValid();
        int getImageDimension();
        int getImageWidth();
        int getImageHeight();
        // Used to approximate good value of w
        int avg_NN_distance();
        std::vector<Image<PixelType>*> getImages();
        ~Dataset();
};


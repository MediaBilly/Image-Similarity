#pragma once

#include <string>
#include <vector>
#include "image.h"
#include "imageReduced.h"

template<typename ImageType, typename PixelType>
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
        int bytes_per_pixel;
        std::vector<ImageType*> images;

    public:
        Dataset(std::string inputPath, int bytes_per_pixel=1);
        bool isValid();
        int getImageDimension();
        // Used to approximate good value of w
        int avg_NN_distance();
        std::vector<ImageType*> getImages();
        ~Dataset();
};


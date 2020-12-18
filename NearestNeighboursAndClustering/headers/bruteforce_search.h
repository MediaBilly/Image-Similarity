#pragma once

#include <vector>
#include <iostream>
#include "image.h"
#include "imageReduced.h"

template<typename ImageType>
class Bruteforce_Search
{
    private:
        std::vector<ImageType*> images;

    public:
        Bruteforce_Search(std::vector<ImageType*> images);
        // Find exact N Nearest Neighbours to query point q
        std::vector<std::pair<double, int>> exactNN(ImageType *q, int N);
        ~Bruteforce_Search();
};

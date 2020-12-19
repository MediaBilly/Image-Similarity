#pragma once

#include <vector>
#include <iostream>
#include "image.h"

template<typename PixelType>
class Bruteforce_Search
{
    private:
        std::vector<Image<PixelType>*> images;

    public:
        Bruteforce_Search(std::vector<Image<PixelType>*> images);
        // Find exact N Nearest Neighbours to query point q
        std::vector<std::pair<double, int>> exactNN(Image<PixelType> *q, int N);
        ~Bruteforce_Search();
};

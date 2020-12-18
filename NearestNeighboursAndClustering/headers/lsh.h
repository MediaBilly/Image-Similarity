#pragma once

#include <vector>
#include "image.h"
#include "dataset.h"
#include "hash_table.h"

class LSH
{
    private:
        int k, w, L;
        std::vector<Image*> images;
        Hash_Table **hashTables;
    public:
        LSH(int k,int w,int L, Dataset<Image,Pixel8Bit> *imageDataset);
        std::vector<std::pair<double, int>> approximate_kNN(Image *q, unsigned int N);
        std::vector<Image*> rangeSearch(Image *q, double r);
        ~LSH();
};
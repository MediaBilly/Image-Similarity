#pragma once

#include <vector>
#include "image.h"
#include "dataset.h"
#include "hash_table.h"

class LSH
{
    private:
        int k, w, L;
        std::vector<Image<Pixel8Bit>*> images;
        Hash_Table **hashTables;
    public:
        LSH(int k,int w,int L, Dataset<Pixel8Bit> *imageDataset);
        std::vector<std::pair<double, int>> approximate_kNN(Image<Pixel8Bit> *q, unsigned int N);
        std::vector<Image<Pixel8Bit>*> rangeSearch(Image<Pixel8Bit> *q, double r);
        ~LSH();
};
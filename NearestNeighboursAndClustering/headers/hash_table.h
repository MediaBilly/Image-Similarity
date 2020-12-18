#pragma once

#include <list>
#include "image.h"
#include "hash_function.h"

class Hash_Table
{
    private:
        int k, w, d, buckets; // d is dimension of the vectors
        Hash_Function *hash_function;
        std::list<Image*> *table; // Array of buckets. Each bucket is a list of images(vectors)
        unsigned long g(Image *image);
    public: 
        Hash_Table(int k, int w, int d, int buckets);  // Constructor 
        std::list<Image*> getBucketImages(Image *q);
        ~Hash_Table();
        bool insert(Image *image);
}; 
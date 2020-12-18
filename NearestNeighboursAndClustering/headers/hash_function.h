#pragma once

// Implements functionality of hash function family H

#include "image.h"

class Hash_Function
{
    private:
        unsigned long k, w, M, d; // d is dimension of the vectors
        double **s;
    public:
        Hash_Function(unsigned long k, unsigned long w, unsigned long d);
        // Calculate's hi
        unsigned long hash(Image *image, unsigned long k);
        ~Hash_Function();
};

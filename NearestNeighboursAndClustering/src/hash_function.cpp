#include <iostream>
#include <random>
#include "../headers/hash_function.h"
#include "../headers/utilities.h"

const unsigned long m = 4294967291; // m = (2^32) - 5


Hash_Function::Hash_Function(unsigned long k, unsigned long w, unsigned long d) {
    // Initialize parameters
    this->k = k;
    this->w = w;
    this->d = d;
    this->M = power(2, 32/k);
    
    // Initialize uniform random distribution number generator
    std::default_random_engine generator;
    std::uniform_real_distribution<double> uniform_distribution(0,this->w - 0.001);

    // For each hash function generate it's Si's, 0 <= i <= d - 1
    this->s = new double*[k];
    for (unsigned long i = 0; i < this->k; i++) {
        this->s[i] = new double[d];
        for (unsigned long j = 0; j < this->d; j++) {
            this->s[i][j] = uniform_distribution(generator);
        }
    }
}

unsigned long Hash_Function::hash(Image *image, unsigned long k) {
    if (k < 0 || k >= this->k) {
        return 0;
    }
    unsigned long cur_hash = 0, cur_m = 1;
    // Calculate current hi (ai's)
    for (int j = this->d - 1; j >= 0; j--) {
        cur_hash = (cur_hash + (cur_m * ((int)floor((image->getPixel(j) - this->s[k][j])/this->w)) % M) % M) % M;
        cur_m = ((m % this->M) * cur_m) % this->M;
    }
    return cur_hash;
}

Hash_Function::~Hash_Function() {
    for (unsigned long i = 0; i < this->k; i++)
        delete[] this->s[i];

    delete[] this->s;
}
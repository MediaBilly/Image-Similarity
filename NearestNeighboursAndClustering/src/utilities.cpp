#include "../headers/utilities.h"

unsigned long power(int base, int exponent) {
    if (exponent == 0)
        return 1;

    unsigned long ret = power(base,exponent >> 1);
    return exponent % 2 == 1 ? ret * ret * base : ret * ret;
}
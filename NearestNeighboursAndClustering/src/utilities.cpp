#include "../headers/utilities.h"
#include <iostream>

unsigned long power(int base, int exponent) {
    if (exponent == 0)
        return 1;

    unsigned long ret = power(base,exponent >> 1);
    return exponent % 2 == 1 ? ret * ret * base : ret * ret;
}


int get_num_from_string(std::string s) {
    std::string num = "";
    int i=0;

    while(isdigit(s[i]))
        num += s[i++];

    return std::stoi(num);
}

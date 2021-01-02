#pragma once

#define SWAP_INT16(x) ((((unsigned short)(x)) >> 8) | (((unsigned short)(x)) << 8))
#define SWAP_INT32(x) ((((unsigned int)(x)) >> 24) | ((((unsigned int)(x)) & 0x00FF0000) >> 8) | ((((unsigned int)(x)) & 0x0000FF00) << 8) | (((unsigned int)(x)) << 24))
#define CEIL(a,b) (((a)+(b)-1)/(b))

// Calculates m^n
unsigned long power(int base, int exponent);
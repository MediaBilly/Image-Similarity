#pragma once

typedef unsigned char Pixel8Bit;
typedef unsigned short Pixel16Bit;

template<typename PixelType>
class Image
{
    private:
        int id;
        int width;
        int height;
        Pixel8Bit *pixels;
    public:
        Image(int id, int width, int height);
        Image(Image &img);
        bool setPixel(int index, PixelType pixel);
        int getId();
        PixelType getPixel(int index);
        int getSize();
        // Calculates the p-norm distance to another image
        double distance(Image *image, int norm);
        ~Image();
};

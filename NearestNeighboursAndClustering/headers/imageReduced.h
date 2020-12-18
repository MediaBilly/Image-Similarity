#pragma once

typedef unsigned short Pixel16Bit;

class ImageReduced
{
    private:
        int id;
        int width;
        int height;
        Pixel16Bit *pixels;
    public:
        ImageReduced(int id, int width, int height);
        ImageReduced(ImageReduced &img);
        bool setPixel(int index, Pixel16Bit pixel);
        int getId();
        Pixel16Bit getPixel(int index);
        int getSize();
        // Calculates the p-norm distance to another image
        double distance(ImageReduced *image, int norm);
        ~ImageReduced();
};

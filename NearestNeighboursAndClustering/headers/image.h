#pragma once

typedef char Pixel8Bit;

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
        bool setPixel(int index, Pixel8Bit pixel);
        int getId();
        Pixel8Bit getPixel(int index);
        int getSize();
        // Calculates the p-norm distance to another image
        double distance(Image *image, int norm);
        ~Image();
};

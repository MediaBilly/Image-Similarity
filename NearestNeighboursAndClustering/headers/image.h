#pragma once

#include <vector>
#include <tuple>

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
        int getWidth();
        int getHeight();
        // Calculates the p-norm distance to another image
        double distance(Image *image, int norm);
        // Sum of all the image's pixel values
        unsigned int totalValue();
        // Normalizes all the pixel values in order the image's totalValue to equal normed_sum
        void normalize(unsigned int normed_sum);

        std::tuple<int,int> findCentroid();

        std::vector<Image<PixelType>*> clusters(int clusterDimension);
        ~Image();
};

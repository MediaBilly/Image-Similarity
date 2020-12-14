from image import Image
import numpy as np
from sklearn.preprocessing import normalize



def createDataset(arr, fileName):
    file = open(fileName, "wb")
    
    file.write((999).to_bytes(4, byteorder='big', signed=False))        # Magic number
    file.write(len(arr).to_bytes(4, byteorder='big', signed=False))     # number of images
    file.write((1).to_bytes(4, byteorder='big', signed=False))          # number of rows
    file.write(len(arr[0]).to_bytes(4, byteorder='big', signed=False))  # number of columns
    
    for image in arr:
        for pixel in image:
            file.write((pixel).to_bytes(2, byteorder='big', signed=False))
    
    file.close()


class ImageDataset:
    def __init__(self, file):
        dataset = open(file, "rb")
        # Read header
        self.magic_num = int.from_bytes(dataset.read(4), byteorder='big', signed=False)
        self.num_of_images = int.from_bytes(dataset.read(4), byteorder='big', signed=False)
        self.num_of_rows = int.from_bytes(dataset.read(4), byteorder='big', signed=False)
        self.num_of_columns = int.from_bytes(dataset.read(4), byteorder='big', signed=False)

        # Read Images
        self.images = []
        for _ in range(self.num_of_images):
            # Create image object
            img = Image(self.num_of_columns, self.num_of_rows)
            # Read pixels of current image
            for _ in range(self.num_of_rows * self.num_of_columns):
                img.addPixel(int.from_bytes(dataset.read(1), byteorder='big', signed=False))

            self.images.append(img)
        
        dataset.close()


    def getImageDimensions(self):
        return (self.num_of_rows, self.num_of_columns)

    def getImagesNormalized(self):
        images = []
        for img in self.images:
            images.append(normalize(img.getPixelsArray(), axis=1, norm='l1'))
            
        return np.array(images)
import numpy as np

class Image:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.pixels = []
        
    
    def getSize(self):
        return self.width * self.height
    
    
    def addPixel(self, pixel):
        self.pixels.append(pixel)
        
        
    def getPixelsArray(self):
        return np.reshape(np.array(self.pixels), (self.width, self.height))
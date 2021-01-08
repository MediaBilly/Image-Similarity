import numpy as np

class LabelDataset:
    def __init__(self, file):
        dataset = open(file, "rb")
        self.classes = set()
        # Read header
        self.magic_num = int.from_bytes(dataset.read(4), byteorder='big', signed=False)
        self.num_of_items = int.from_bytes(dataset.read(4), byteorder='big', signed=False)
        # Read Images
        self.labels = []
        for _ in range(self.num_of_items):
            label = int.from_bytes(dataset.read(1), byteorder='big', signed=False)
            if label not in self.classes:
                self.classes.add(label)
            self.labels.append(label)
        
        dataset.close()

    def get_labels(self):
        return np.array(self.labels)

    def num_classes(self):
        return len(self.classes)
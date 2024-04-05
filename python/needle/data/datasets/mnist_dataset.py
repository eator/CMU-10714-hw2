from typing import List, Optional
from ..data_basic import Dataset
import numpy as np
import gzip
import struct

def parse_mnist(image_filename, label_filename):
    # read image file
    with gzip.open(image_filename, 'rb') as fi:
        image_content = fi.read()

    image_magic = struct.unpack('>I', image_content[:4])[0]
    if image_magic != 2051 :
        print("read file format error!")

    image_num = struct.unpack('>I', image_content[4:8])[0]
    image_row = struct.unpack('>I', image_content[8:12])[0]
    image_col = struct.unpack('>I', image_content[12:16])[0]

    # Reshape image_content into a 3D array (image_num, image_row, image_col)
    image_content_reshaped = np.frombuffer(image_content, dtype=np.uint8, offset=16).reshape(image_num, image_row, image_col)
    # Convert uint8 values to float32 and normalize
    X = image_content_reshaped.astype(np.float32) / 255.0
    # Reshape X into a 2D array (image_num, image_row * image_col)
    X = X.reshape(image_num, -1)
    # --------------------------------------------------------------------------------------#

    # read label file
    with gzip.open(label_filename, 'rb') as fl:
        label_content = fl.read()

    label_magic = struct.unpack('>I', label_content[:4])[0]
    if label_magic != 2049 :
        print("read file format error!")

    label_num = struct.unpack('>I', label_content[4:8])[0]
    y = np.zeros(label_num, dtype=np.uint8)

    # get y from label_content
    for i in range (label_num):
        label = label_content[8 + i]
        y[i] = np.uint8(label)
    # --------------------------------------------------------------------------------------#

    return (X, y, image_row, image_col)

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.image, self.label, self.row, self.col = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index) -> object:
        ### BEGIN YOUR SOLUTION
        X = self.image[index]
        y = self.label[index]
        m = self.row
        n = self.col

        X_in = X.reshape((m,n,-1))
        X_out = self.apply_transforms(X_in)

        return X_out.reshape(-1, m*n), y
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.label.shape[0]
        ### END YOUR SOLUTION

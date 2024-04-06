import numpy as np
import needle as ndl
import math
from ..autograd import Tensor

from typing import Iterator, Optional, List, Sized, Union, Iterable, Any



class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
    
    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x


class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
     """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size

        if not self.shuffle:
            self.ordering = np.array_split(np.arange(len(dataset)), 
                                        range(batch_size, len(dataset), batch_size))

    def __iter__(self):
        ### BEGIN YOUR SOLUTION
        if self.shuffle:
            shuffle_order = np.arange(len(self.dataset))
            np.random.shuffle(shuffle_order)
            self.ordering = np.array_split(shuffle_order,
                                    range(self.batch_size, len(self.dataset), self.batch_size))

        self.it = 0
        return self
        ### END YOUR SOLUTION

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __next__(self):
        ### BEGIN YOUR SOLUTION
        if self.it >= len(self):
            raise StopIteration

        idx = self.it
        self.it += 1

        # TODO to be figured out
        return tuple([Tensor(x) for x in self.dataset[self.ordering[idx]]]) 

        #types = len(tuple(self.dataset[0]))
        ##print(types)
        #batch_data = []

        #for i in range(self.batch_size):
        #    #print("&&&&&&")
        #    #print(self.dataset[self.ordering[self.it][i]][0].shape)
        #    items = list(self.dataset[self.ordering[self.it][i]])
        #    #print(Tensor(data).shape)
        #    if len(batch_data) == 0:
        #        for x in items:
        #            if isinstance(self.dataset, ndl.data.NDArrayDataset):
        #                newshape = (1,) + x.shape
        #                x = x.reshape(newshape)

        #            if np.isscalar(x):
        #                batch_data.append([x]) 
        #            else:
        #                batch_data.append(x) 

        #        #print(Tensor(batch_data).shape)
        #    else:
        #        #print(Tensor(batch_data).shape)
        #        #print(Tensor(data).shape)
        #        for i, x in enumerate(items):
        #            #print("******")
        #            #print(Tensor(batch_data[i]).shape)
        #            if isinstance(self.dataset, ndl.data.NDArrayDataset):
        #                newshape = (1,) + x.shape
        #                x = x.reshape(newshape)

        #            if np.isscalar(x):
        #                batch_data[i] = np.concatenate((batch_data[i], [x]))
        #            else:
        #                batch_data[i] = np.concatenate((batch_data[i], x))
        #            #print("&&&&&&")
        #            #print(Tensor(batch_data[i]).shape)
        #            #print('\n')

        #self.it += 1        

        #return tuple([Tensor(x) for x in batch_data]) 
        ### END YOUR SOLUTION

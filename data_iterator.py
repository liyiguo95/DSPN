import pandas as pd
import numpy as np
import random
class dataIterator:
    def __init__(self, source,
                 batch_size=32,
                 shuffle_each_epoch=True,
    ):
        random.seed(233)
        self.source = source
        self.shuffle = shuffle_each_epoch
        if self.shuffle:
            random.shuffle(self.source)
        self.batch_size = batch_size
        self.count = 0
        self.total = len(self.source)
    
    def reset(self):
        self.count = 0
        if self.shuffle:
            random.shuffle(self.source)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.count == self.total:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        raw_data = []
        self.tmp = 0
        while self.count < self.total:
            raw_data.append(self.source[self.count])
            self.count += 1
            self.tmp += 1
            if self.tmp >= self.batch_size:
                break
        if len(raw_data) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration
        return raw_data
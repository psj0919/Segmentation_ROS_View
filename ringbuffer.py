import numpy as np

class CircularBuffer:
    def __init__(self, max_size=10):
        self.cell = np.zeros((640,480,3), dtype='float32')
        self.buffer = [self.cell] * max_size

        self.head = 0
        self.tail = 0
        self.numItems = 0
        self.max_size = max_size
    
    def __str__(self):
        items = ['{!r}'.format(item) for item in self.buffer]
        return '['+', '.join(items)+']'

    def size(self):
        if self.tail >= self.head:
            return self.tail - self.head
        return self.max_size - self.head - self.tail
    
    def is_empty(self):
        return self.numItems == 0

    def is_full(self):
        return self.numItems == self.max_size

    def enqueue(self, item):
        if self.is_full():
            raise OverflowError("CircularBuffer is full, unable to enqueue item")
        self.buffer[self.head] = item
        self.head = (self.head+1) % self.max_size
        self.numItems += 1

    def front(self):
        return self.buffer[self.tail]

    def dequeue(self):
        if self.is_empty():
            raise IndexError("CircularBuffer is empty, unable to dequeue")
        item = self.buffer[self.tail]
        self.buffer[self.tail] = None
        self.tail = (self.tail + 1) % self.max_size
        self.numItems -= 1

        return item

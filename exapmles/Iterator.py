"""
Простейший итератор, без использования абстрактных классов
"""


class Counter:
    def __init__(self, low, high):
        self.current = low + 1
        self.high = high

    def __iter__(self):
        return self

    def __next__(self):
        self.current += 1
        if self.current < self.high:
            return self.current
        raise StopIteration


for i in Counter(1, 10):
    print(i)

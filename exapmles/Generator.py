"""
Простейший генератор
"""
import time


def gen():
    while 1:
        print("До")
        yield 1
        print("После")


g = gen()





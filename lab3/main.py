import pygame as pg
from keras.models import Sequential
from keras.layers import Dense

width, height = 500, 500

if __name__ == "__main__":
	pg.init()
	display = pg.display.set_mode((width, height))
	clock = pg.time.clock()

	
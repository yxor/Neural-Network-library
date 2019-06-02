from nn import NeuraNetwork
import random

from math import sin, cos, pi

import pygame as pg
import sys
import os


# initing pygame
CAPTION = "learning to draw a circle"
WIDTH = 700
HEIGHT = 700
SCREEN_SIZE = (WIDTH, HEIGHT)
BACKGROUND_COLOR = pg.Color("darkslategrey")
os.environ['SDL_VIDEO_CENTERED'] = '1'
pg.init()
pg.display.set_caption(CAPTION)
pg.display.set_mode(SCREEN_SIZE)
screen = pg.display.get_surface()
screen_rect = screen.get_rect()
center_x, center_y = WIDTH/2, HEIGHT/2


sin_brain = NeuraNetwork(1, 32, 1)
cos_brain = NeuraNetwork(1, 32, 1)
#sin_brain.load("sin")
#cos_brain.load("cos")

def make_dataset(number, function):
	dataset = []
	for _ in range(number):
		inputs = [random.uniform(0, 2 * pi)]
		outputs = [(function(inputs[0])+1)/2]
		dataset.append([inputs,outputs])
	return dataset

def train():
    sin_data = make_dataset(2000, sin)
    cos_data = make_dataset(2000, cos)

    for data in sin_data:
        sin_brain.backward(data[0], data[1])

    for data in cos_data:
        cos_brain.backward(data[0], data[1])


r = 200
done = False
while not done:
    screen.fill(BACKGROUND_COLOR)
    for event in pg.event.get():
        if event.type == pg.QUIT:
            cos_brain.save("cos")
            sin_brain.save("sin")
            done = True
    
    # drawing the real circle
    for _ in range(1000):
        angle = random.uniform(0, 2 * pi)
        x = cos(angle) * r + center_x
        y = sin(angle) * r + center_y
        pg.draw.circle(screen, (255, 0, 0), (int(x), int(y)), 5)
    
    # drawing the scuffed one
    for _ in range(1000):
        angle = random.uniform(0, 2 * pi)
        x = (cos_brain.feedForward([angle])[0]*2 -1) * r + center_x
        y = (sin_brain.feedForward([angle])[0]*2 -1) * r + center_y
        pg.draw.circle(screen, (0, 0, 0), (int(x), int(y)), 5)
    

    train()
    pg.display.update()


pg.quit()
sys.exit()
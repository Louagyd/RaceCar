import pygame
from pygame.locals import *
import numpy as np
import math
import matplotlib.pyplot as plt
import pickle as pkl
import os
from util import Map


pygame.init()
width, height = 500, 500
screen=pygame.display.set_mode((width, height))

pygame.mixer.init()

pen = pygame.image.load("res/image/pen.png")
redpen = pygame.image.load("res/image/redpen.png")
car = pygame.image.load("res/image/big_car2.png")
pensize = [20,20]
carsize = [50,50]

starting_poses = []
starting_thetas= []
theta = 270/360*2*math.pi

drawed = np.zeros(shape=[500, 500, 3])
is_drawing = False
draw_red = False
draw_car = False
running = True
while running:
    screen.fill(0)
    surf = pygame.surfarray.make_surface(drawed)
    screen.blit(surf, (0,0))
    for i in range(len(starting_poses)):
        carthispos = [starting_poses[i][0] - carsize[0]/2, starting_poses[i][1] - carsize[1]/2]
        this_theta = starting_thetas[i]
        carthis_area = this_theta%(math.pi/2)
        carthis_scale = math.cos(carthis_area)*(1+math.tan(carthis_area))
        diflen_carthis = (carthis_scale-1)*(carsize[0]/(math.sqrt(2)))
        carthisrot = pygame.transform.rotate(car, this_theta/2/math.pi*360 - 270)
        screen.blit(carthisrot, (carthispos[0] - diflen_carthis/math.sqrt(2), carthispos[1] - diflen_carthis/math.sqrt(2)))
    if draw_red:
        redpenscale = pygame.transform.scale(redpen, pensize)
        screen.blit(redpenscale, (pygame.mouse.get_pos()[0] - pensize[0]/2, pygame.mouse.get_pos()[1] - pensize[1]/2))
    elif draw_car:
        car2rot = pygame.transform.rotate(car, theta/2/math.pi*360 - 270)
        car2pos = [pygame.mouse.get_pos()[0]-carsize[0]/2, pygame.mouse.get_pos()[1]-carsize[1]/2]
        car2_area = theta%(math.pi/2)
        car2_scale = math.cos(car2_area)*(1+math.tan(car2_area))
        diflen_car2 = (car2_scale-1)*(carsize[0]/(math.sqrt(2)))
        screen.blit(car2rot, (car2pos[0] - diflen_car2/math.sqrt(2), car2pos[1] - diflen_car2/math.sqrt(2)))
    else:
        penscale = pygame.transform.scale(pen, pensize)
        screen.blit(penscale, (pygame.mouse.get_pos()[0] - pensize[0]/2, pygame.mouse.get_pos()[1] - pensize[1]/2))
    if is_drawing:
        if not draw_red and not draw_car:
            drawed[int(pygame.mouse.get_pos()[0] - pensize[0]/2):int(pygame.mouse.get_pos()[0] + pensize[0]/2), int(pygame.mouse.get_pos()[1] - pensize[1]/2):int(pygame.mouse.get_pos()[1] + pensize[1]/2), :] = 255
        elif draw_red:
            drawed[int(pygame.mouse.get_pos()[0] - pensize[0]/2):int(pygame.mouse.get_pos()[0] + pensize[0]/2), int(pygame.mouse.get_pos()[1] - pensize[1]/2):int(pygame.mouse.get_pos()[1] + pensize[1]/2), 0] = 150
            drawed[int(pygame.mouse.get_pos()[0] - pensize[0]/2):int(pygame.mouse.get_pos()[0] + pensize[0]/2), int(pygame.mouse.get_pos()[1] - pensize[1]/2):int(pygame.mouse.get_pos()[1] + pensize[1]/2), 1:3] = 0

    pygame.display.flip()
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            # if it is quit the game
            pygame.quit()
            exit(0)
        if event.type == pygame.MOUSEBUTTONDOWN:
            if draw_car:
                starting_poses.append([pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]])
                starting_thetas.append(theta)
            else:
                is_drawing = True
        if event.type == pygame.MOUSEBUTTONUP:
            is_drawing = False
        if event.type == pygame.KEYDOWN:
            if event.key == K_r:
                draw_car = False
                draw_red = True
            if event.key == K_c:
                draw_red = False
                draw_car = True
            if event.key == K_w:
                draw_red = False
                draw_car = False
            if event.key == K_n:
                theta -= 0.1
            if event.key == K_m:
                theta += 0.1
            if event.key == K_x:
                pensize[0] += 3
                pensize[1] += 3
            if event.key == K_z:
                if (pensize[0] > 3):
                    pensize[0] -= 3
                    pensize[1] -= 3
            if event.key == K_q:
                drawed = np.zeros([500,500,3])
                starting_poses = []
                starting_thetas = []
            if event.key == K_s:
                if os.path.exists("Maps/Images/"):
                    num_acc = len(os.listdir("Maps/Images"))
                else:
                    num_acc = 0
                    os.makedirs("Maps/Images/")
                plt.imsave("Maps/Images/Map" + str(num_acc+1) + ".jpg", drawed/255)
                if os.path.exists("Maps/data.pkl"):
                    the_data = pkl.load( open( "Maps/data.pkl", "rb" ) )
                else:
                    the_data = []
                this_map = Map(drawed, starting_poses, starting_thetas)
                the_data.append(this_map)
                pkl.dump(the_data, open( "Maps/data.pkl" , "wb" ) )
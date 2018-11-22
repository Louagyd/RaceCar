import pygame
from pygame.locals import *
from DQN3 import DQN
from scipy import misc
import random
import math
import tensorflow as tf
import util
import numpy as np
import os
import shutil
import sys
from PIL import Image
import imageio
from util import Map
import pickle as pkl

pygame.init()

width, height = 500, 500

car_num = 2
init_lambda = 0.005
lambd = init_lambda
num_frames = 1

screen=pygame.display.set_mode((width, height))

sess = tf.Session()

car_size = [50, 50]
window_size = 200
margin = int((35/100) * window_size)

pygame.mixer.init()

# field = misc.imread('res/image/arena.png')
# field = field[:,:,0]
# bg = pygame.image.load("res/image/arena.png")
car = pygame.image.load("res/image/big_car.png")
car2 = pygame.image.load("res/image/big_car2.png")

theta_array = [math.pi/8, 2*math.pi/8, 3*math.pi/8, 4*math.pi/8, 5*math.pi/8,  6*math.pi/8,  7*math.pi/8]

global starting_poses
global starting_thetas
starting_poses = []
starting_thetas= []
theta2 = 270/360*2*math.pi

scenario = []
records = []

agent = DQN(len(theta_array), 5, [6], 0.001, 32, 0.97, 0.99, 5000, False)
agent.load_model(sess, 'Models/Driver2')

frames = np.zeros(shape=[window_size, window_size, num_frames]) + 255
old_frames=frames

global field
field = np.zeros([500,500,3])
field_use = np.zeros([500+2*window_size, 500+2*window_size]) + 255


global pos_list
global theta_list
global v_list
v_mean = 5

def new_scene():
    the_data = pkl.load(open("Maps/data.pkl", "rb"))
    map_idx = np.random.randint(0, len(the_data))
    this_map = the_data[map_idx]
    global field
    field = this_map.drawing
    global starting_poses
    global starting_thetas
    starting_poses = this_map.starting_poses
    starting_thetas = this_map.starting_thetas
    global pos_list
    global theta_list
    global v_list
    pos_list = []
    theta_list = []
    v_list = []
    the_scene = util.Scene(field, pos_list, theta_list, v_list)
    return the_scene

scene = new_scene()
scenario.append(scene)

running = True
rect = pygame.Rect(0,0,500,500)
sub = screen.subsurface(rect)
records.append(sub)
accident = False
to_remove = []
while running:
    screen.fill(0)
    surf = pygame.surfarray.make_surface(field)
    screen.blit(surf, (0,0))


    car2rot = pygame.transform.rotate(car2, theta2/2/math.pi*360 - 270)

    car2pos = [pygame.mouse.get_pos()[0]-car_size[0]/2, pygame.mouse.get_pos()[1]-car_size[1]/2]
    car2_area = theta2%(math.pi/2)
    car2_scale = math.cos(car2_area)*(1+math.tan(car2_area))
    diflen_car2 = (car2_scale-1)*(car_size[0]/(math.sqrt(2)))
    screen.blit(car2rot, (car2pos[0] - diflen_car2/math.sqrt(2), car2pos[1] - diflen_car2/math.sqrt(2)))

    for i in range(len(starting_poses)):
        carthispos = [starting_poses[i][0] - car_size[0]/2, starting_poses[i][1] - car_size[1]/2]
        this_theta = starting_thetas[i]
        carthis_area = this_theta%(math.pi/2)
        carthis_scale = math.cos(carthis_area)*(1+math.tan(carthis_area))
        diflen_carthis = (carthis_scale-1)*(car_size[0]/(math.sqrt(2)))
        carthisrot = pygame.transform.rotate(car2, this_theta/2/math.pi*360 - 270)
        screen.blit(carthisrot, (carthispos[0] - diflen_carthis/math.sqrt(2), carthispos[1] - diflen_carthis/math.sqrt(2)))

    for iter in range(len(pos_list)):
        this_rel_pos = [pos_list[iter][0], height - pos_list[iter][1]]
        this_carrot = pygame.transform.rotate(car, theta_list[iter]/2/math.pi*360 - 270)
        this_carpos = [this_rel_pos[0]-car_size[0]/2, this_rel_pos[1]-car_size[1]/2]
        this_car_area = theta_list[iter]%(math.pi/2)
        this_car_scale = math.cos(this_car_area)*(1+math.tan(this_car_area))
        this_diflen_car = (this_car_scale-1)*(car_size[0]/(math.sqrt(2)))
        screen.blit(this_carrot, (this_carpos[0] - this_diflen_car/math.sqrt(2), this_carpos[1] - this_diflen_car/math.sqrt(2)))

        field2 = np.transpose(field[:,:,0].copy())
        # field2 = np.transpose(field2)
        for pos in pos_list:
            if pos != pos_list[iter]:
                rel_pos = [500 - pos[1], pos[0]]
                field2[(int(rel_pos[0])-20):(int(rel_pos[0])+20), (int(rel_pos[1])-20):(int(rel_pos[1])+20)] = 0
        field_use[window_size:(500+window_size), window_size:(500+window_size)] = field2
        view = util.crop_from_array(field_use, [pos_list[iter][0]+window_size, pos_list[iter][1]+window_size], theta_list[iter], window_size, margin)
        # view = np.flip(view, axis=0)
        view = np.flip(view, axis=1)
        for i in range(num_frames-1):
            frames[:,:,i] = frames[:,:,i+1]
        frames[0:view.shape[0],0:view.shape[1],num_frames-1] = view

        this_dists, acc_points = util.get_dists(frames[:,:,0], [int(window_size/2), int(window_size/2) + margin], int(window_size/2), int(window_size/10), theta_array)

        action = agent.choose_action(this_dists)
        if action == 0:
            r = 0.02
        if action == 1:
            r = 0.01
        if action == 2:
            r = 0
        if action == 3:
            r = -0.01
        if action == 4:
            r = -0.02

        new_x = pos_list[iter][0] + v_list[iter]*math.cos(theta_list[iter])
        new_y = pos_list[iter][1] + v_list[iter]*math.sin(theta_list[iter])
        theta_list[iter] += r*v_list[iter]
        theta_list[iter] = theta_list[iter] % (2*math.pi)
        pos_list[iter] = [new_x, new_y]
        if field2[int(this_rel_pos[1]), int(this_rel_pos[0])] < 100:
            accident = True

        elif field2[int(this_rel_pos[1]), int(this_rel_pos[0])] < 200:
            to_remove.append(iter)

    for remove_idx in sorted(to_remove, reverse=True):
        del pos_list[remove_idx]
        del theta_list[remove_idx]
        del v_list[remove_idx]

    to_remove = []

    assert len(v_list) == len(pos_list)
    assert len(v_list) == len(theta_list)

    for iteration in range(len(starting_poses)):
        if len(pos_list) > 0:
            nearest = min(pos_list, key=lambda x:abs((x[0]-starting_poses[iteration][0])**2 + (x[1] - (height-starting_poses[iteration][1]))**2))
            min_dist = (nearest[0]-starting_poses[iteration][0])**2 + (nearest[1] - (height-starting_poses[iteration][1]))**2
        else:
            min_dist = 100000

        if min_dist > 10000:
            the_rand = random.random()
            if the_rand < lambd:
                pos_list.append([starting_poses[iteration][0], height - starting_poses[iteration][1]])
                theta_list.append(starting_thetas[iteration] + random.random()*math.pi/4 - math.pi/8)
                v_list.append(random.random()*v_mean + v_mean/2)
    lambd += 0.00002
    if accident:
        if len(scenario) > 20:
            num_acc = len(os.listdir("Results/Dangerous"))
            num2 = len(os.listdir("Results/Temp"))

            im1 = Image.open("Results/Temp/" + "Temp"+ str(num2 - 20) + ".jpg")
            im2 = Image.open("Results/Temp/" + "Temp"+ str(num2 - 10) + ".jpg")
            im3 = Image.open("Results/Temp/" + "Temp"+ str(num2 - 1) + ".jpg")
            new_im = Image.new('RGB', (3*width, height))

            new_im.paste(im1, (0,0))
            new_im.paste(im2, (width,0))
            new_im.paste(im3, (2*width,0))

            new_im.save('Results/Dangerous/Accident' + str(num_acc+1) + ".jpg")
            images_for_gif = []
            for this_num in range(20):
                le_image = imageio.imread("Results/Temp/Temp" + str(num2 - 20 + this_num) + ".jpg")
                images_for_gif.append(le_image)
            imageio.mimsave("Results/Dangerous_Gifs/Gif" + str(num_acc+1) + ".gif", images_for_gif, fps = 5)

        records = []
        scenario = []
        pos_list = []
        theta_list = []
        v_list = []
        accident = False
        scene = new_scene()
        scenario.append(scene)
        lambd = init_lambda

        if os.path.exists("Result/Temp"):
            shutil.rmtree("Results/Temp")

    sub = screen.subsurface(rect)
    if os.path.exists("Results/Temp/"):
        num_acc = len(os.listdir("Results/Temp"))
    else:
        num_acc = 0
        os.makedirs("Results/Temp/")

    pygame.image.save(sub, "Results/Temp/Temp" + str(num_acc) + ".jpg")

    records.append(sub)
    scene = util.Scene(field, pos_list, theta_list, v_list)
    scenario.append(scene)


    pygame.display.flip()
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            # if it is quit the game
            pygame.quit()
            exit(0)

        if event.type == pygame.MOUSEBUTTONDOWN:
            starting_poses.append([pygame.mouse.get_pos()[0], pygame.mouse.get_pos()[1]])
            starting_thetas.append(theta2)

        if event.type == pygame.KEYDOWN:
            if event.key == K_r:
                scene = new_scene(field, car_num)
                scenario = []
                records = []
                shutil.rmtree("Results/Temp")

            if event.key == K_n:
                theta2 -= 0.1
            if event.key == K_m:
                theta2 += 0.1
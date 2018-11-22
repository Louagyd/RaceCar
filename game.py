import pygame
from pygame.locals import *
from scipy import misc
import math
import util
import numpy as np
import random

# from DQN import DQN
from DQN3 import DQN

pygame.init()
width, height = 650, 500
car_size = [20, 20]
window_size = 100
handle_size = 100
margin = 35
theta_array = [math.pi/8, 2*math.pi/8, 3*math.pi/8, 4*math.pi/8, 5*math.pi/8,  6*math.pi/8,  7*math.pi/8]
# theta_array = [math.pi/2]
screen=pygame.display.set_mode((width, height))

pygame.mixer.init()

field = misc.imread('res/image/field.png')
field = field[:,:,0]
field_use = np.zeros([500+2*window_size, 500+2*window_size]) + 255
field_use[window_size:(500+window_size), window_size:(500+window_size)] = field

pygame.mixer.music.load('res/audio/unfaithful.mp3')
pygame.mixer.music.play(-1, 0.0)
pygame.mixer.music.set_volume(0.25)

bg = pygame.image.load("res/image/field.png")
car = pygame.image.load("res/image/car.png")
car2 = pygame.image.load("res/image/car2.png")
handle = pygame.image.load("res/image/handle.png")
handle_bg = pygame.image.load("res/image/handlebg.png")
velbg = pygame.image.load("res/image/velbg.png")
velpoint = pygame.image.load("res/image/velpoint.png")

starting_poses = []
starting_thetas= []
pos = [100,250]
pos_player = [100,250]
theta = 270/360*2*math.pi
theta_player = 270/360*2*math.pi
theta2 = 270/360*2*math.pi
starting_poses.append(pos)
starting_thetas.append(theta)
init_v = 3
v = init_v
v_player = init_v
r = 0
r_player = 0
score = 0
last_score = 0
max_score = 0
num_frames = 1

layers = []
layers.append({'num_filters':32,
               'kernel_size':[8,8],
               'stride_size':[4,4]})
layers.append({'num_filters':64,
               'kernel_size':[4,4],
               'stride_size':[2,2]})
layers.append({'num_filters':64,
               'kernel_size':[3,3],
               'stride_size':[2,2]})

# agent = DQN([window_size, window_size, num_frames], 5, layers, 512, 1e-6, 32, 0.97, 0.99, 5000, False)
agent = DQN(len(theta_array), 5, [6], 0.001, 32, 0.97, 0.99, 5000, False)

frames = np.zeros(shape=[window_size, window_size, num_frames])+255
old_frames=frames
dists = [0] * len(theta_array)
old_dists = dists
reward = 0
action = 0
action_player = 2

player_is_playing = True
hide = False
shuffle_start = True
done = False
done_player = False
running = 1
ct = 0
while running:
    ct += 1
    score += (v-score)/100
    screen.fill(0)

    screen.blit(bg, (0,0))

    carrot = pygame.transform.rotate(car, theta/2/math.pi*360 - 270)
    carrot_player = pygame.transform.rotate(car2, theta_player/2/math.pi*360 - 270)
    car2rot = pygame.transform.rotate(car2, theta2/2/math.pi*360 - 270)
    handlerot = pygame.transform.rotate(handle, 5*r/2/math.pi*360)
    handlepos = [500, window_size]
    handle_area = 5*r%(math.pi/2)
    handle_scale = math.cos(handle_area)*(1+math.tan(handle_area))
    diflen_handle = (handle_scale-1)*(handle_size/(math.sqrt(2)))
    screen.blit(handle_bg, (500, window_size))
    screen.blit(handlerot, (500 - diflen_handle/math.sqrt(2), window_size - diflen_handle/math.sqrt(2)))
    screen.blit(velbg, (500, window_size + handle_size))
    screen.blit(velpoint, (500 + handle_size/5 + v*8 - 10, window_size + handle_size + 11))

    if not hide:
        rel_pos = [pos[0], 500 - pos[1]]

        carpos = [rel_pos[0]-car_size[0]/2, rel_pos[1]-car_size[1]/2]
        car_area = theta%(math.pi/2)
        car_scale = math.cos(car_area)*(1+math.tan(car_area))
        diflen_car = (car_scale-1)*(car_size[0]/(math.sqrt(2)))
        screen.blit(carrot, (carpos[0] - diflen_car/math.sqrt(2), carpos[1] - diflen_car/math.sqrt(2)))

    if player_is_playing:
        rel_pos_player = [pos_player[0], 500 - pos_player[1]]
        carpos_player = [rel_pos_player[0]-car_size[0]/2, rel_pos_player[1]-car_size[1]/2]
        car_area_player = theta_player%(math.pi/2)
        car_scale_player = math.cos(car_area_player)*(1+math.tan(car_area_player))
        diflen_car_player  = (car_scale_player-1)*(car_size[0]/(math.sqrt(2)))
        screen.blit(carrot_player, (carpos_player[0] - diflen_car_player/math.sqrt(2), carpos_player[1] - diflen_car_player/math.sqrt(2)))

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

    view = util.crop_from_array(field_use, [pos[0]+window_size, pos[1]+window_size], theta, window_size, margin)
    # view = np.flip(view, axis=0)
    view = np.flip(view, axis=1)

    for iter in range(num_frames-1):
        frames[:,:,iter] = frames[:,:,iter+1]
    frames[:,:,num_frames-1] = 255
    frames[0:view.shape[0],0:view.shape[1],num_frames-1] = view

    dists, acc_points = util.get_dists(frames[:,:,0], [int(window_size/2), int(window_size/2) + margin], 50, 10, theta_array)

    # agent.watch_and_learn(old_frames, action, reward, frames)
    if not hide:
        agent.watch_and_learn(old_dists, action, reward, dists)

    surf = pygame.surfarray.make_surface(view)
    screen.blit(surf, (500,0))

    for the_point in acc_points:
        pygame.draw.line(screen, (255,200,200), [int(window_size/2) + 500, int(window_size/2) + margin], [the_point[0] + 500, the_point[1]])

    screen.blit(pygame.transform.flip(car, False, True), (500 + window_size/2 - car_size[0]/2, window_size/2 - car_size[1]/2 + margin))

    if r > 0.06:
        r = 0.06
    if r < -0.06:
        r= -0.06
    if v > 20:
        v = 20
    if v < -20:
        v = -20

    pygame.font.init()
    font = pygame.font.Font(None, 24)
    text = font.render("Score: "+str(int(score*10)), True, (255,0,0))
    screen.blit(text, (500, window_size + handle_size + 80))
    pygame.font.init()
    font = pygame.font.Font(None, 24)
    text = font.render("Last: "+str(int(last_score*10)), True, (255,0,0))
    screen.blit(text, (500, window_size + handle_size + 100))
    font = pygame.font.Font(None, 24)
    text = font.render("Max: "+str(int(max_score*10)), True, (255,0,0))
    screen.blit(text, (500, window_size + handle_size + 120))

    frames = (frames/255)*2 - 1

    # action = agent.choose_action(frames)
    action = agent.choose_action(dists)
    # action = 2

    # if action == 0:
    #     v += 0.1
    # if action == 1:
    #     v -= 0.1
    if action == 0:
        r = 0.04
    if action == 1:
        r = 0.02
    if action == 2:
        r = 0
    if action == 3:
        r = -0.02
    if action == 4:
        r = -0.04


    if action_player == 0:
        r_player = 0.02
    if action_player == 1:
        r_player = 0.01
    if action_player == 2:
        r_player = 0
    if action_player == 3:
        r_player = -0.01
    if action_player == 4:
        r_player = -0.02



    new_x = pos[0] + v*math.cos(theta)
    new_y = pos[1] + v*math.sin(theta)
    theta += r*v
    pos = [new_x, new_y]

    new_x = pos_player[0] + v*math.cos(theta_player)
    new_y = pos_player[1] + v*math.sin(theta_player)
    theta_player += r_player*v_player
    pos_player = [new_x, new_y]

    reward = v/10
    rel_pos = [pos[0], 500 - pos[1]]
    if field[int(rel_pos[1]), int(rel_pos[0])] < 100:
        done = True
        reward = -1


    rel_pos_player = [pos_player[0], 500 - pos_player[1]]
    if field[int(rel_pos_player[1]), int(rel_pos_player[0])] < 100:
        done_player = True

    old_frames = frames
    old_dists = dists
    if done:
        if shuffle_start:
            the_rand = int(np.random.randint(0, len(starting_poses), 1))
            pos = [starting_poses[the_rand][0], 500 - starting_poses[the_rand][1]]
            theta = starting_thetas[the_rand]
        v = random.random()*init_v + init_v/2
        r = 0
        done = False
        if score > max_score:
            max_score = score
        last_score = score
        score = 0

    if done_player:
        pos_player = [starting_poses[0][0], 500 - starting_poses[0][1]]
        theta_player = starting_thetas[0]
        v_player = init_v
        r_player = 0
        done_player = False





    if ct % 1000 == 0:
        print(ct)

    # 7 - update the screen
    pygame.display.flip()
    for event in pygame.event.get():
        # check if the event is the X button
        if event.type==pygame.QUIT:
            # if it is quit the game
            pygame.quit()
            exit(0)

        if event.type == pygame.MOUSEBUTTONDOWN:
            starting_poses.append(pygame.mouse.get_pos())
            starting_thetas.append(theta2)

        if event.type == pygame.KEYDOWN:
            if event.key == K_a:
                if action_player > 2:
                    action_player   = 1
                else:
                    action_player = 0
            if event.key == K_s:
                action_player = 2
            if event.key == K_d:
                if action_player < 3:
                    action_player = 3
                else:
                    action_player = 4

            if event.key == K_n:
                theta2 -= 0.1
            if event.key == K_m:
                theta2 += 0.1

            if event.key == K_z:
                init_v -= 1
                v_player -= 1
                v -= 1
            if event.key == K_x:
                init_v += 1
                v_player += 1
                v += 1

            if event.key == K_p:
                if shuffle_start:
                    shuffle_start = False
                else:
                    shuffle_start = True

            if event.key == K_b:
                if player_is_playing:
                    player_is_playing = False
                else:
                    player_is_playing = True

            if event.key == K_h:
                if hide:
                    hide = False
                else:
                    hide = True

            if event.key == K_r:
                max_score = 0
            if event.key == K_q:
                done = True
            if event.key == K_t:
                agent.start_training()
                print("ok")
            if event.key == K_i:
                print(agent.last_q)

            if event.key == K_o:
                agent.save_model('Models/Driver3')

        # if event.type == pygame.KEYUP:
        # v += 1
        # if event.type==pygame.MOUSEBUTTONDOWN:
        #     shoot.play()
        #     position=pygame.mouse.get_pos()
        #     acc[1]+=1
        #     arrows.append([math.atan2(position[1]-(playerpos1[1]+32),position[0]-(playerpos1[0]+26)),playerpos1[0]+32,playerpos1[1]+32])
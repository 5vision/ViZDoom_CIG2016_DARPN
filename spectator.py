#!/usr/bin/env python

from vizdoom import *

import numpy as np
import cv2
import os
import cPickle
import sys

track_num = int(sys.argv[1])
map_num = int(sys.argv[2])
ticks = int(sys.argv[3])
bots = int(sys.argv[4])

epochs = 10
episodes = 1

width = 64
height = 48


def preprocess(image, width, height):
    image = cv2.resize(image[0], (width, height), interpolation=cv2.INTER_AREA)
    return image.reshape((1, 1, height, width)).astype(np.float32) / 255.0

def backup(batch, track_num, map_num):
    folder = 'data_track{}/'.format(track_num)
    title = 'm{}_t{}_b{}'.format(map_num, ticks, bots)
    count = len([name for name in os.listdir(folder) if title in name])
    name = '{}{}_{:02d}.pkl'.format(folder, title, count)
    with open(name, 'wb') as file:
        cPickle.dump(batch, file)
        print name, len(batch)

game = DoomGame()
game.load_config("train.cfg")
game.set_doom_map('map{:02d}'.format(map_num))  # 1 - Limited deathmatch, 2 - Full deathmatch

game.set_episode_timeout(12600) # 6 minutes
game.set_window_visible(True)

game.set_mode(Mode.SPECTATOR)
game.add_game_args("-host 1 -deathmatch +timelimit 10.0 +name 5vision +colorset 5 "
                   "+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1")
game.init()

for epoch in range(epochs):
    
    batch = []

    for episode in range(episodes):
        
        path = []

        game.new_episode()

        game.send_game_command("removebots")
        for bot in range(bots):
            game.send_game_command("addbot")

        while not game.is_episode_finished():

            s = game.get_state()
            img_prev = preprocess(s.image_buffer, width, height)
            var_prev = s.game_variables

            game.advance_action(ticks)
            a = game.get_last_action()

            """
            w = [0]*6
            w[0] = int(game.get_game_variable(GameVariable.AMMO2) * game.get_game_variable(GameVariable.WEAPON2))
            w[1] = int(game.get_game_variable(GameVariable.AMMO3) * game.get_game_variable(GameVariable.WEAPON3))
            w[2] = int(game.get_game_variable(GameVariable.AMMO4) * game.get_game_variable(GameVariable.WEAPON4))
            w[3] = int(game.get_game_variable(GameVariable.AMMO5) * game.get_game_variable(GameVariable.WEAPON5))
            w[4] = int(game.get_game_variable(GameVariable.AMMO6) * game.get_game_variable(GameVariable.WEAPON6))
            w[5] = int(game.get_game_variable(GameVariable.AMMO7) * game.get_game_variable(GameVariable.WEAPON7)) - 40

            best_weapon = 1
            for i, v in reversed(list(enumerate(w))):
                if v > 0:
                    best_weapon = i + 2
                    break

            selected_weapon = game.get_game_variable(GameVariable.SELECTED_WEAPON)

            if selected_weapon != best_weapon:
                print 'choose weapon', best_weapon
            """

            path.append((img_prev, var_prev, a))

            if game.is_player_dead() or game.is_episode_finished():

                var_next = []
                var_next.append(game.get_game_variable(GameVariable.FRAGCOUNT))
                var_next.append(game.get_game_variable(GameVariable.HEALTH))
                var_next.append(game.get_game_variable(GameVariable.ARMOR))
                var_next.append(game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO))
                var_next.append(game.get_game_variable(GameVariable.SELECTED_WEAPON))
                
                path.append((None, var_next, None))
                batch.append(path)

                path = []

                game.respawn_player()

    backup(batch, track_num, map_num)

game.close()

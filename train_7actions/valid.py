#!/usr/bin/env python

from vizdoom import *
from utils import *

from policy import PolicyTest

import cv2
import numpy as np
import time
import os, glob
import sys
import yaml

# Load config
with open(sys.argv[1], 'r') as ymlfile:
    config = yaml.load(ymlfile)

track_num = int(sys.argv[2]) # 1 - Limited deathmatch, 2 - Full deathmatch

n_bots = 5

def validate_model(file_name):

    policy = PolicyTest(file_name, config)

    game = DoomGame()
    game.load_config('train.cfg')
    game.set_doom_map('map{:02d}'.format(track_num))
    game.set_window_visible(False)
    game.set_episode_timeout(35 * 60 * 10) # 10 minutes
    game.set_mode(Mode.PLAYER)
    game.add_game_args('-host 1 -deathmatch +timelimit 10.0 +name 5vision +colorset 5 '
                       '+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1')
    game.init()
    
    f = []
    
    for _ in range(30):

        frags_last = 0
        u_idx = 6

        policy.reset_state()
        game.new_episode()

        game.send_game_command('removebots')
        for bot in range(n_bots):
            game.send_game_command('addbot')

        while not game.is_episode_finished():

            s = game.get_state()

            o = preprocess(s.image_buffer, policy.width, policy.height)

            p = policy.output(u_idx, o)

            u_idx, u_lst = create_action(p[0], True)

            game.make_action(u_lst, 2)

            frags_curr = game.get_game_variable(GameVariable.FRAGCOUNT)
            f.append(frags_curr - frags_last)
            frags_last = frags_curr

            if game.is_player_dead():        
                game.respawn_player()
                policy.reset_state()
                u_idx = 6

    f = np.array(f)
    print file_name, f[f>0].sum(), f[f<0].sum(), f.sum()

for file in glob.glob("models/policy_*_track{}.pkl".format(track_num)):
    validate_model(file)
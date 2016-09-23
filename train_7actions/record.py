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
    
net_name = sys.argv[2]
track_num = int(sys.argv[3]) # 1 - Limited deathmatch, 2 - Full deathmatch

# Clear previous game
filelist = glob.glob('images/*.png')
for f in filelist:
    os.remove(f)
print 'Removed {} previous files'.format(len(filelist))


is_visible = False


file_name = 'models/policy_{}_{}_track{}.pkl'.format(net_name, config['name'], track_num)

policy = PolicyTest(file_name, config)

width = policy.width
height = policy.height

game = DoomGame()
game.load_config('train.cfg')
game.set_doom_map('map{:02d}'.format(track_num))
game.set_episode_timeout(5000)
game.set_window_visible(is_visible)
game.set_mode(Mode.PLAYER)
game.add_game_args('-host 1 -deathmatch +timelimit 10.0 +name 5vision +colorset 5 '
                   '+sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1')
game.init()

policy.reset_state()
game.new_episode()

n_bots = 5

game.send_game_command('removebots')
for bot in range(n_bots):
    game.send_game_command('addbot')

t = 0
f = []
frags_last = 0
u_idx = 6

while not game.is_episode_finished():

    s = game.get_state()
    
    o = preprocess(s.image_buffer, width, height)

    p = policy.output(u_idx, o)
    
    if not is_visible:      
        
        att = policy._deconv_attention(policy.attention)
        
        att = att.reshape((height, width, 1)) * 255
        att = cv2.resize(att, (320, 240), interpolation=cv2.INTER_CUBIC)
        
        obs = s.image_buffer.reshape((240, 320, 1))
        att = att.reshape((240, 320, 1))
        img = np.concatenate((obs, att), axis=1)
        
        cv2.imwrite('images/{:03d}.png'.format(t), img)

    u_idx, u_lst = create_action(p[0], True)

    game.make_action(u_lst, 2)
    
    t += 1

    if is_visible:
        time.sleep(0.01)

    frags_curr = game.get_game_variable(GameVariable.FRAGCOUNT)
    f.append(frags_curr - frags_last)
    frags_last = frags_curr

    if game.is_player_dead():        
        game.respawn_player()
        policy.reset_state()
        u_idx = 6
    
f = np.array(f)

print 'Reward:', f[f>0].sum(), f[f<0].sum(), f.sum()

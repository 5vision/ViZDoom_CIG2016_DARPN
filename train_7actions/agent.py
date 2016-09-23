from vizdoom import *
from utils import *

import numpy as np
import multiprocessing
from multiprocessing import Queue


game_args = '-host 1 -deathmatch +timelimit 10.0 +name 5vision +colorset 5 +sv_forcerespawn 1 +sv_noautoaim 1 +sv_respawnprotect 1 +sv_spawnfarthest 1'

# colors: 0 - green, 1 - gray, 2 - brown, 3 - red, 4 - light gray, 5 - light brown, 6 - light red, 7 - light blue


def evaluation(track_num, policy):
    
    width = 64
    height = 48
        
    game_test = DoomGame()
    game_test.load_config('train.cfg')
    game_test.set_mode(Mode.PLAYER)
    game_test.set_doom_map('map{:02d}'.format(track_num))
    game_test.add_game_args(game_args)
    game_test.set_window_visible(False)
    game_test.set_episode_timeout(35 * 60 * 10) # 10 minutes
    game_test.init()
    f = []
    for _ in range(5):
        frags = 0
        
        policy.reset_state()
        
        game_test.new_episode()
        
        game_test.send_game_command('removebots')
        for bot in range(5):
            game_test.send_game_command('addbot')
        
        u_idx = 6
            
        while not game_test.is_episode_finished():
            s = game_test.get_state()
            o = preprocess(s.image_buffer, width, height)
            p = policy.output(u_idx, o)
            u_idx, u_lst = create_action(p[0])
            game_test.make_action(u_lst, 2)
            frags_curr = game_test.get_game_variable(GameVariable.FRAGCOUNT)
            f.append(frags_curr - frags)
            frags = frags_curr
            if game_test.is_player_dead():        
                game_test.respawn_player()
                policy.reset_state()
                u_idx = 6
        
    game_test.close()
    return np.array(f)


class Agent(multiprocessing.Process):
    
    def __init__(self, al, track_num=1):
        multiprocessing.Process.__init__(self)
        
        self.al = al
        self.track_num = track_num
        
        self.width = 64
        self.height = 48
        self.history = 50

        self.actions_queue = Queue()
        self.states_queue = Queue()
        self.data_queue = Queue()
        self.rnd = np.random.RandomState()
        
    def run(self):
        
        states = []
        actions = []
        rewards = []
        frags = []
        
        bot_id = int(self.name.split('-')[1])
        
        map_num = bot_id % 2
        
        if self.track_num == 2 and bot_id > 2:
            map_num += 2
            
        doom_map = 'map{:02d}'.format(map_num)
        
        game = DoomGame()
        game.load_config('train.cfg')
        game.set_mode(Mode.PLAYER)
        game.set_doom_map(doom_map)
        game.add_game_args(game_args)
        game.set_window_visible(False)
        game.init()
        
        n_bots = 7 # only for track2 else 5
        
        game.send_game_command('removebots')
        for bot in range(n_bots):
            game.send_game_command('addbot')
        
        print '%s: start at %s' % (self.name, doom_map)
        
        u_idx = 6
        
        while True:
            
            s = game.get_state()
            img_prev = s.image_buffer
            var_prev = s.game_variables
            
            obs_prev = preprocess(img_prev, self.width, self.height)
            
            self.states_queue.put((u_idx, obs_prev))
            
            p = self.actions_queue.get()
            
            if p is None:
                break
            
            u_idx, u_lst = create_action(p, False, self.rnd)
            game.make_action(u_lst, 2)

            # If a player is dead or an episode is finished the state will be None,
            # but we need these variables to calculate a last reward.
            var_next = []
            var_next.append(game.get_game_variable(GameVariable.FRAGCOUNT))
            var_next.append(game.get_game_variable(GameVariable.HEALTH))
            var_next.append(game.get_game_variable(GameVariable.ARMOR))
            var_next.append(game.get_game_variable(GameVariable.SELECTED_WEAPON_AMMO))
            var_next.append(game.get_game_variable(GameVariable.SELECTED_WEAPON))
            
            var_state = self.al.get_state(var_prev, var_next)
            
            states.append(obs_prev)
            actions.append(u_idx)
            rewards.append(self.al.get_reward(var_state))
            frags.append(var_next[0] - var_prev[0])
            
            # TODO: add deadpool cases
            
            reset = False
            data = None
            
            if game.is_player_dead() or game.is_episode_finished():
                
                if len(states) > self.history and var_next[1] > -300 and max(frags) != min(frags):
                    S = np.concatenate(states).astype(np.float32)
                    A = np.array(actions).astype(np.int32)
                    R = np.array(rewards).astype(np.float32)
                    F = np.array(frags).astype(np.int32)
                    data = (S, A, R, F)
                
                u_idx = 6
                
                del states[:]
                del actions[:]
                del rewards[:]
                del frags[:]
                
                game.new_episode()
                game.send_game_command('removebots')
                for bot in range(n_bots):
                    game.send_game_command('addbot')
                
                reset = True
            
            self.data_queue.put((reset, data))

        game.close()
        return
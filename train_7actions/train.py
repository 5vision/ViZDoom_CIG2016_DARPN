#!/usr/bin/env python

from vizdoom import *

from utils import discount_cumsum
from baseline import BaselineLearner
from policy import PolicyLearner
from reward import ApprenticeshipLearner

from agent import Agent, evaluation

import numpy as np
import time

import sys
import yaml

with open(sys.argv[1], 'r') as ymlfile:
    config = yaml.load(ymlfile)

history = config['history']
batch_size = config['batch_size']

track_num = int(sys.argv[2]) # 1 - Limited deathmatch, 2 - Full deathmatch

discount = 0.995
gae = 0.97


al = ApprenticeshipLearner()

N = 10
agents = []
for n in range(N):
    agents.append(Agent(al, track_num))
for agent in agents:
    agent.start()
    

baseline = BaselineLearner(config)

policy = PolicyLearner(config)


# Load pretrained weights

#al.restore()

suffix_name = '_{}_track{}.pkl'.format(config['name'], track_num)

policy.restore('models/policy_pretrain_last' + suffix_name)
baseline.copy_gru(policy.gru)


# Continue training by NGD

epoch = 0
max_frags = 0
max_reward = 0
epoch_reward = 0


uprevs = np.empty(N, np.int32)
obs = np.empty((N, 1, policy.height, policy.width), np.float32)
hiddens1 = np.repeat(policy.gru.h01.get_value(), N, 0)
hiddens2 = np.repeat(policy.gru.h02.get_value(), N, 0)
attentions = np.repeat(policy.gru.a0.get_value(), N, 0)


for i in range(5000):
    
    start_time = time.time()
    
    V = []
    S = []
    Y = []
    A = []
    R = []
    F = []
    average_length = []
    average_reward = []

    while len(S) < batch_size:
        
        for n in range(N):
            u, o = agents[n].states_queue.get()
            uprevs[n] = u
            obs[n] = o
        
        attentions, hiddens1, hiddens2, p = policy._output_step(uprevs, obs, attentions, hiddens1, hiddens2)
        
        for n in range(N):
            agents[n].actions_queue.put(p[n])
        
        for n in range(N):
            reset, data = agents[n].data_queue.get()
            
            if reset is True:
                attentions[n] = policy.gru.a0.get_value()
                hiddens1[n] = policy.gru.h01.get_value()
                hiddens2[n] = policy.gru.h02.get_value()
                
            if data is not None:
                
                states, actions, rewards, frags = data

                returns = discount_cumsum(rewards, discount)

                v = baseline._output(actions, states)

                # The generalized advantage estimator (GAE)
                deltas = rewards + discount * np.append(v[1:],v[-1]) - v
                advantages = discount_cumsum(deltas, discount * gae)

                length = len(states)
                m = min(length // history, 3)
                w = length // m
                for j in range(m):
                    offset = np.random.randint(j * w, (j + 1) * w - history + 1)
                    V.append(v[offset:offset+history])
                    S.append(states[offset:offset+history])
                    Y.append(returns[offset:offset+history])
                    A.append(actions[offset:offset+history])
                    R.append(advantages[offset:offset+history])

                F.append(frags)
                average_length.append(length)
                average_reward.append(rewards.sum())
    
    num_samples = len(S)
    F = np.concatenate(F)
    average_length = np.array(average_length).mean()
    average_reward = np.array(average_reward).mean()
    
    if average_reward >= max_reward:
        max_reward = average_reward
        policy.backup('models/policy_best_reward' + suffix_name)
    
    # Step 1. Estimate reward function
    #al.learn(batch)
    
    # TODO: it's possible to use the new reward function
    
    # Step 2. Train baseline
    A = np.concatenate(A).astype(np.int32)
    V = np.concatenate(V).astype(np.float32)
    S = np.concatenate(S).astype(np.float32)
    Y = np.concatenate(Y).astype(np.float32)
    
    baseline_loss = baseline.train(A, V, S, Y)
    #if (i + 1) % 100 == 1:
    #    baseline.debug(A, V, S, Y)
    del V, Y
    
    # Step 3. Train policy
    
    # With RNN scheme KL divergence increases due to a different h_0,
    # thus we feedforward filtered states again. 
    # Also we must be carefull with deterministic=False.
    
    P = policy._output_batch(A, S)
    
    R = np.concatenate(R).astype(np.float32)
    # Standardize advantage
    R = (R - R.mean()) / R.std()
    
    policy_loss = policy.train(S, P, A, R)
    #if (i + 1) % 100 == 1:
    #    policy.debug(S, P, A, R)
    del S, P, A, R

    print '{:3d}>  T {:2d}/{:}  F {:2d}/{: 3d}/{: 3d}  R {:.2f}  L {:.4f}/{: .4f}  N {:.1f}/{:.1f}  E {:.2f}  LM {:.3f}  KL {:.3f}  RR {:.3f}  AC {:2d}  BT {:.3f}/{}/{}'.format(
        i + 1, # iteration
        num_samples, int(average_length),# number of samples/length of trajectories
        F[F>0].sum(), F[F<0].sum(), F.sum(), # frags
        average_reward,
        float(baseline_loss[0]),
        float(policy_loss[0]),
        float(baseline_loss[1]), # Norm of baseline's gradient before clipping
        float(policy_loss[8]), # Norm of riemannian gradients
        float(policy_loss[3]), # Entropy
        float(policy_loss[1]), # Lagrange multiplier
        float(policy_loss[2]) * 10000, # KL divergence,
        float(policy_loss[4]) * 1000, # Relative residual
        int(policy_loss[5]),   # A condition
        float(policy_loss[6]), # Ratio
        policy_loss[7],        # Number of backtracks
        int(time.time() - start_time)
    )
    
    epoch_reward += average_reward
    
    if (i + 1) % 20 == 0:
        
        epoch += 1
        
        f = evaluation(track_num, policy)
        
        test_frags = f.sum()
        
        loginfo = 'Epoch: {:2d} frags {:2d}/{: 3d}/{: 3d}  reward {:.2f}\n'.format(
            epoch, f[f>0].sum(), f[f<0].sum(), test_frags, epoch_reward / 20
        )
        
        print loginfo
        with open('train_track{}.log'.format(track_num), 'a+') as logfile:
            logfile.write(loginfo)
        
        #max_reward = 0.9 * max_reward
        epoch_reward = 0
        
        if test_frags >= max_frags:
            max_frags = test_frags
            policy.backup('models/policy_best_frags' + suffix_name)
        
        baseline.backup('models/baseline_last' + suffix_name)
        policy.backup('models/policy_last' + suffix_name)

for n in range(N):
    agents[n].actions_queue.put(None)
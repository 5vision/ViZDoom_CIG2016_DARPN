#!/usr/bin/env python

import numpy as np
from random import shuffle

import theano

from utils import parse_action
from policy import PolicyPretrain
from reward import ApprenticeshipLearner

import os
import cPickle

import sys
import yaml

import time

with open(sys.argv[1], 'r') as ymlfile:
    config = yaml.load(ymlfile)

history = config['history']

track_num = int(sys.argv[2]) # 1 - Limited deathmatch, 2 - Full deathmatch

policy = PolicyPretrain(config)


def restore(track_num=1, only_test=False, file_mask=None):
    data = []
    i = 0
    total_ticks = 0
    folder = '../data_track{}/'.format(track_num)
    for file_name in os.listdir(folder):
        
        if file_mask is not None and file_mask not in file_name:
            continue
        
        if ('test' in file_name) != only_test:
            continue
            
        with open(folder+file_name, 'rb') as data_file:
            batch = cPickle.load(data_file)
        
        num_frames = 0
        for path in batch:
            num_frames += len(path) - 1
        if '_t2_' in file_name:
            total_ticks += num_frames * 2
        else:
            total_ticks += num_frames
        print '{:2d} {}: {:2d} trajectories, {:5d} frames, {}'.format(
            i, file_name, len(batch), num_frames, total_ticks
        )
        data.extend(batch)
        i += 1
    print 'Total ticks: {}, time: {:.2f} hours'.format(total_ticks, float(total_ticks) / (35 * 60 * 60))
    return data


def convert(chunk):
    S = []
    A = []
    for path in chunk:
        
        states = []
        actions = []
        
        for t in range(len(path) - 1):            
            s = path[t][0]
            a = parse_action(path[t][2])            
            states.append(s)
            actions.append(a)
            
        length = len(states)
        
        if length > history:
            n = min(length // history, 5)
            w = length // n
            for i in range(n):
                # Try to balance a dataset
                for j in range(3):
                    offset = np.random.randint(i * w, (i + 1) * w - history + 1)
                    a = actions[offset:offset+history]
                    if 0 in a or np.random.rand() > 0.7:
                        S.extend(states[offset:offset+history])
                        A.extend(a)
                        break
            
    S = np.concatenate(S).astype(np.float32)
    A = np.array(A).astype(np.int32)
    return S, A


# Test dataset
test = restore(track_num, True)
test_s, test_a = convert(test)
del test[:]
print 'Test dataset:', test_s.shape

# Train dataset
train = restore(track_num, False)

if track_num == 2:
    train.extend(restore(1, False, '_t2_'))

# Split data into chunks, due to a memory constraint
chunk_size = 250 if len(train) > 250 else len(train)
print 'Train dataset: {}/{}'.format(len(train), chunk_size)

#al = ApprenticeshipLearner()
#al.add_expert(train)
#al.backup()

loss_best = 1.0

num_epochs = 500

lr = 0.001
lr_min = 0.00001
lr_decay = float(lr - lr_min) / num_epochs

for epoch in range(num_epochs):
    
    start_time = time.time()
    
    lr = np.cast[theano.config.floatX](max(lr - lr_decay, lr_min))

    shuffle(train)

    updates = 0
    loss_train = 0

    for offset in range(0, len(train), chunk_size):
        if offset+chunk_size > len(train):
            break
        chunk = train[offset:offset+chunk_size]
        s, a = convert(chunk)
        u, l = policy.pretrain(s, a, lr)
        updates += u
        loss_train += l
        
    loss_test = policy.validate(test_s, test_a)

    loginfo = 'Epoch {:2d} lr {:.5f} updates {} time {} loss {:.3f}/{:.3f}'.format(
        epoch, float(lr), updates, int(time.time() - start_time), loss_train / updates, loss_test
    )
    
    print loginfo
    with open('pretrain_track{}.log'.format(track_num), 'a+') as logfile:
        logfile.write(loginfo+'\n')

    if loss_test <= loss_best:
        loss_best = loss_test
        policy.backup('models/policy_pretrain_best_{}_track{}.pkl'.format(config['name'], track_num))
    policy.backup('models/policy_pretrain_last_{}_track{}.pkl'.format(config['name'], track_num))
#!/usr/bin/python

from __future__ import print_function
from vizdoom import *

import cPickle

import cv2

import numpy as np

import theano
import theano.tensor as T

import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.init as LI


np.random.seed(4516)
    
def preprocess(image, width=64, height=48):
    image = cv2.resize(image[0], (width, height), interpolation=cv2.INTER_AREA)
    return image.reshape((1, 1, height, width)).astype(np.float32) / 255.0

def create_action(p):
    c = np.cumsum(p)
    a = c.searchsorted(np.random.uniform(0, c[-1]))
    
    if a == 0 and p[0] < 0.1:
        a = np.argmax(p)
    
    if a == 0:
        return a, [1,0,0,0] # ATTACK
    if a == 1:
        return a, [0,1,0,1] # TURN_LEFT and MOVE_FORWARD
    if a == 2:
        return a, [0,0,1,1] # TURN_RIGHT and MOVE_FORWARD
    if a == 3:
        return a, [0,1,0,0] # TURN_LEFT
    if a == 4:
        return a, [0,0,1,0] # TURN_RIGHT
    if a == 5:
        return a, [0,0,0,1] # MOVE_FORWARD
    return a, [0,0,0,0] # NOP

def relu(x):
    return T.maximum(x, 0)

def build_cnn(l_input, use_noise=False, use_bn=True):
    
    # NOTE: Neither Conv2DDNNLayer nor Conv2DMMLayer will not work
    # with T.Rop operation used for the Fisher-vector product.

    l_out = L.Conv2DLayer(l_input,
        num_filters=32, filter_size=(6,6), stride=2,
        nonlinearity=relu, W=LI.HeUniform('relu'), b=LI.Constant(0.)
    )
    
    # https://arxiv.org/pdf/1602.01407v2.pdf
    # QUOTE: KFC-pre and BN can be combined synergistically.
    
    if use_bn: l_out = L.batch_norm(l_out, beta=None, gamma=None)

    l_out = L.Conv2DLayer(l_out,
        num_filters=32, filter_size=(4,4), stride=2,
        nonlinearity=relu, W=LI.HeUniform('relu'), b=LI.Constant(0.)
    )
    
    if use_bn: l_out = L.batch_norm(l_out, beta=None, gamma=None)
    if use_noise: l_out = L.dropout(l_out)
    
    l_out = L.Conv2DLayer(l_out,
        num_filters=64, filter_size=(4,4), stride=2,
        nonlinearity=relu, W=LI.HeUniform('relu'), b=LI.Constant(0.)
    )
    
    if use_bn: l_out = L.batch_norm(l_out, beta=None, gamma=None)
    if use_noise: l_out = L.dropout(l_out)
    
    return l_out

# https://github.com/rllab/rllab/blob/master/rllab/core/network.py
class GRUStepLayer(L.MergeLayer):

    def __init__(self, incomings, num_units, att_units, num_actns,
                 zero_trainable=True, zero_init=LI.Constant(0.),
                 W_init=LI.HeUniform(), b_init=LI.Constant(0.), s_init=LI.Constant(1.)):

        super(GRUStepLayer, self).__init__(incomings, name=None)

        # Output of CNN has shape [batch * step, channel, height, width]
        cnn_shape = self.input_shapes[1]
        num_chann = cnn_shape[1]
        num_pixel = cnn_shape[2] * cnn_shape[3]
        
        # Weights for the initial hidden state and attention map
        self.a0 = self.add_param(zero_init, (1, num_pixel), name="a0",
                                 trainable=zero_trainable,
                                 regularizable=False)
        self.h01 = self.add_param(zero_init, (1, num_units), name="h01",
                                 trainable=zero_trainable,
                                 regularizable=False)
        self.h02 = self.add_param(zero_init, (1, num_units), name="h02",
                                 trainable=zero_trainable,
                                 regularizable=False)
        
        # Weights for the attention gate
        self.W_ha = self.add_param(W_init, (num_units, att_units), name="W_ha")
        self.W_aa = self.add_param(W_init, (num_pixel, att_units), name="W_aa")
        self.W_ua = self.add_param(W_init, (num_actns, att_units), name="W_ua")
        self.W_xa = self.add_param(W_init, (num_chann, att_units), name="W_xa")
        self.b_p = self.add_param(b_init, (att_units,), name="b_p", regularizable=False)
        self.W_pa = self.add_param(W_init, (att_units,1), name="W_pa")
        self.b_a = self.add_param(b_init, (1,), name="b_a", regularizable=False)  
        
        # Weights for the reset/update gate of 1 layer
        self.W_xg = self.add_param(W_init, (num_chann, num_units*2), name="W_xg")
        self.W_hg = self.add_param(W_init, (num_units, num_units*2), name="W_hg")
        self.b_g = self.add_param(b_init, (num_units*2,), name="b_g", regularizable=False)
        # Weights for the cell gate of 1 layer
        self.W_xc = self.add_param(W_init, (num_chann, num_units), name="W_xc")
        self.W_hc = self.add_param(W_init, (num_units, num_units), name="W_hc")
        self.b_c = self.add_param(b_init, (num_units,), name="b_c", regularizable=False)      
        
        # Weights for the reset/update gate of 2 layer
        self.W2_xg = self.add_param(W_init, (num_units, num_units*2), name="W2_xg")
        self.W2_hg = self.add_param(W_init, (num_units, num_units*2), name="W2_hg")
        self.b2_g = self.add_param(b_init, (num_units*2,), name="b2_g", regularizable=False)
        # Weights for the cell gate of 2 layer
        self.W2_xc = self.add_param(W_init, (num_units, num_units), name="W2_xc")
        self.W2_hc = self.add_param(W_init, (num_units, num_units), name="W2_hc")
        self.b2_c = self.add_param(b_init, (num_units,), name="b2_c", regularizable=False)
        
        self.attention = None
        self.hidden1 = None
        
        self.num_units = num_units
        self.num_pixel = num_pixel
        self.num_chann = num_chann
        self.step_shape = (-1, num_chann, num_pixel)

    def step(self, x, xprj, aprev, hprev, h2prev):
        
        #############################################################
        # Attention gate, RNN version http://arxiv.org/abs/1607.05108
        #############################################################
        
        # Input x [batch, height * width, channel]
        
        a = T.dot(hprev, self.W_ha) + T.dot(aprev, self.W_aa)
        
        a = T.tanh(a[:,None,:] + xprj)
        
        # [batch, height * width, att_units]
        
        a = T.dot(a, self.W_pa) + self.b_a
        
        a = a.reshape((-1, self.num_pixel))
        
        a = T.nnet.softmax(a) # [batch, height * width]
        
        x = (x * a[:,:,None]).sum(1)
        
        # Output x [batch, channel]
        
        #############################################################
        # Recurrent gates of 1 layer
        #############################################################
        
        xg = x.dot(self.W_xg)
        hg = hprev.dot(self.W_hg)
        
        g = T.nnet.sigmoid(xg + hg + self.b_g)
        
        reset = g[:,:self.num_units]
        update = g[:,self.num_units:]
        
        xc = x.dot(self.W_xc)
        hc = hprev.dot(self.W_hc)
        
        c = T.tanh(xc + reset * hc + self.b_c)
        
        h = (1 - update) * hprev + update * c
        
        #############################################################
        # Recurrent gates of 2 layer
        #############################################################
        
        xg = h.dot(self.W2_xg)
        hg = h2prev.dot(self.W2_hg)
        
        g = T.nnet.sigmoid(xg + hg + self.b2_g)
        
        reset = g[:,:self.num_units]
        update = g[:,self.num_units:]
        
        xc = h.dot(self.W2_xc)
        hc = h2prev.dot(self.W2_hc)
        
        c = T.tanh(xc + reset * hc + self.b2_c)
        
        h2 = (1 - update) * h2prev + update * c
        
        return [a, h, h2]

    def get_output_shape_for(self, input_shapes):
        n_batches = input_shapes[0][0]
        return n_batches, self.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        
        uprev, x, aprev, hprev, h2prev = inputs
        
        # x [batch, channel, height, width]
        
        x = x.reshape(self.step_shape)
        
        shuffled_x = x.dimshuffle(0, 2, 1)
        projected_x = T.dot(shuffled_x, self.W_xa) + self.b_p
        
        
        uprev = self.W_ua[uprev]
        projected_x = projected_x + uprev[:,None,:]
        
        rval = self.step(shuffled_x, projected_x, aprev, hprev, h2prev)
        
        self.attention = rval[0]
        self.hidden1 = rval[1]
        
        return rval[2]

def create_network(n_actions, file_name, width=64, height=48, gru_units=64, att_units=48):
    
    l_action = L.InputLayer((None,))
    l_input = L.InputLayer((None, 1, height, width))
    l_attention = L.InputLayer((None, 24))
    l_hidden1 = L.InputLayer((None, gru_units))
    l_hidden2 = L.InputLayer((None, gru_units))

    l_cnn = build_cnn(l_input)

    l_gru = GRUStepLayer([l_action, l_cnn, l_attention, l_hidden1, l_hidden2],
                         gru_units, att_units, n_actions)

    l_out = L.DenseLayer(l_gru, num_units=n_actions, nonlinearity=LN.softmax)
    
    with open(file_name, 'rb') as file:
        L.set_all_param_values(l_out, cPickle.load(file))

    action = T.ivector('action')
    state = T.tensor4('state')
    attention = T.matrix('attention')
    hidden1 = T.matrix('hidden1')
    hidden2 = T.matrix('hidden2')

    step_hidden2, step_output = L.get_output(
        [l_gru, l_out],
        {l_action:action, l_input:state, l_attention:attention, l_hidden1:hidden1, l_hidden2:hidden2},
        deterministic=True
    )
    
    step_hidden1 = l_gru.hidden1
    step_attention = l_gru.attention
    
    _output_step = theano.function(
        [action, state, attention, hidden1, hidden2],
        [step_attention, step_hidden1, step_hidden2, step_output]
    )

    return l_gru, _output_step


l_gru, _output_step = create_network(7, 'policy_final_track1.pkl')

attention = l_gru.a0.get_value()
hidden1 = l_gru.h01.get_value()
hidden2 = l_gru.h02.get_value()
u_idx = 6

game = DoomGame()
game.load_config("5vision.cfg")
game.add_game_args("+name 5vision +colorset 5")
game.init()

#t = 0

while not game.is_episode_finished():

    if game.is_player_dead():
        game.respawn_player()
        attention = l_gru.a0.get_value()
        hidden1 = l_gru.h01.get_value()
        hidden2 = l_gru.h02.get_value()
        u_idx = 6
        
    s = game.get_state()

    action = np.array([u_idx], np.int32)
    state = preprocess(s.image_buffer)
    
    #cv2.imwrite("images/{:03d}.png".format(t), state.reshape((48, 64, 1))*255)
    #t += 1
    
    attention, hidden1, hidden2, p = _output_step(
        action, state, attention, hidden1, hidden2
    )
    
    u_idx, u_lst = create_action(p[0])

    game.make_action(u_lst, 2)

    # Log your frags every ~5 seconds
    if s.number % 175 == 0:
        print("Frags:", game.get_game_variable(GameVariable.FRAGCOUNT))

game.close()

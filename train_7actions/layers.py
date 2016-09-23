import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import lasagne.nonlinearities as LN
import lasagne.init as LI


# https://github.com/TimSalimans/weight_norm/blob/master/nn.py
# T.nnet.relu has some issues with very large inputs, this is more stable
def relu(x):
    return T.maximum(x, 0)

def build_cnn(config, use_noise=True, use_bn=True):
    
    # NOTE: Neither Conv2DDNNLayer nor Conv2DMMLayer will not work
    # with T.Rop operation, which used for the Fisher-vector product.
    
    l_input = L.InputLayer((None, 1, config['height'], config['width']))

    l_out = L.Conv2DLayer(l_input,
        num_filters=config['cnn_f1'], filter_size=(6,6), stride=2,
        nonlinearity=relu, W=LI.HeUniform('relu'), b=LI.Constant(0.)
    )
    
    # https://arxiv.org/pdf/1602.01407v2.pdf
    # QUOTE: KFC-pre and BN can be combined synergistically.
    
    if use_bn: l_out = L.batch_norm(l_out, beta=None, gamma=None)

    l_out = L.Conv2DLayer(l_out,
        num_filters=config['cnn_f2'], filter_size=(4,4), stride=2,
        nonlinearity=relu, W=LI.HeUniform('relu'), b=LI.Constant(0.)
    )
    
    if use_bn: l_out = L.batch_norm(l_out, beta=None, gamma=None)
    if use_noise: l_out = L.dropout(l_out)
    
    l_out = L.Conv2DLayer(l_out,
        num_filters=config['cnn_f3'], filter_size=(4,4), stride=2,
        nonlinearity=relu, W=LI.HeUniform('relu'), b=LI.Constant(0.)
    )
    
    if use_bn: l_out = L.batch_norm(l_out, beta=None, gamma=None)
    if use_noise: l_out = L.dropout(l_out)
    
    return l_input, l_out

def build_deconv(l_input):
    l_input = L.ReshapeLayer(l_input, ([0], 1, 4, 6))
    l_out = L.TransposedConv2DLayer(l_input,
        num_filters=1, filter_size=(4,4), stride=2,
        nonlinearity=None, W=LI.Constant(0.4), b=LI.Constant(0.)
    )
    l_out = L.TransposedConv2DLayer(l_out,
        num_filters=1, filter_size=(4,4), stride=2,
        nonlinearity=None, W=LI.Constant(0.4), b=LI.Constant(0.)
    )
    l_out = L.TransposedConv2DLayer(l_out,
        num_filters=1, filter_size=(6,6), stride=2,
        nonlinearity=None, W=LI.Constant(0.4), b=LI.Constant(0.)
    )
    return l_out

# https://github.com/rllab/rllab/blob/master/rllab/core/network.py
class GRULayer(L.MergeLayer):

    def __init__(self, incomings, num_steps, num_units, att_units, num_actns=7,
                 zero_trainable=True, zero_init=LI.Constant(0.),
                 W_init=LI.HeUniform(), b_init=LI.Constant(0.), s_init=LI.Constant(1.)):

        super(GRULayer, self).__init__(incomings, name=None)

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
        self.num_steps = num_steps
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

    def get_path_layer(self, incomings):
        return GRUPathLayer(incomings, gru_layer=self)
    
    def get_step_layer(self, incomings):
        return GRUStepLayer(incomings, gru_layer=self)

    def get_output_shape_for(self, input_shapes):
        n_batch_steps = input_shapes[0][0]
        return n_batch_steps, self.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        
        uprevs, xs = inputs
        
        # xs [batch * step, channel, height, width]
        
        n_batches = xs.shape[0] / self.num_steps
        
        xs = T.reshape(xs, (n_batches, self.num_steps, self.num_chann, -1))
        
        shuffled_xs = xs.dimshuffle(1, 0, 3, 2)
        projected_xs = T.dot(shuffled_xs, self.W_xa) + self.b_p
        
        
        unops = T.alloc(6, n_batches, 1)
        uprevs = T.reshape(uprevs, (n_batches, self.num_steps))
        uprevs = T.concatenate([unops, uprevs[:,:-1]], axis=1)
        uprevs = self.W_ua[uprevs].dimshuffle(1, 0, 2)
        projected_xs = projected_xs + uprevs[:,:,None,:]
        
        
        a0s = T.repeat(self.a0, n_batches, 0)
        h01s = T.repeat(self.h01, n_batches, 0)
        h02s = T.repeat(self.h02, n_batches, 0)
        
        rval, _ = theano.scan(
            fn=self.step,
            sequences=[shuffled_xs, projected_xs],
            outputs_info=[a0s, h01s, h02s]
        )
        
        hs = rval[2].dimshuffle(1, 0, 2)
        
        return hs.reshape((-1, self.num_units))


class GRUPathLayer(L.MergeLayer):
    def __init__(self, incomings, gru_layer, name=None):
        super(GRUPathLayer, self).__init__(incomings, name)
        self._gru = gru_layer

    def get_params(self, **tags):
        return self._gru.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        n_steps = input_shapes[0][0]
        return n_steps, self._gru.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        
        uprevs, xs = inputs
        
        # xs [step, channel, height, width]
        
        xs = xs.reshape(self._gru.step_shape)
        
        shuffled_xs = xs.dimshuffle(0, 2, 1)
        projected_xs = T.dot(shuffled_xs, self._gru.W_xa) + self._gru.b_p
        
        
        unop = T.alloc(6, 1)
        uprevs = T.concatenate([unop, uprevs[:-1]])
        uprevs = self._gru.W_ua[uprevs]
        projected_xs = projected_xs + uprevs[:,None,:]
        
        
        # convert from row to matrix
        a0s = T.unbroadcast(self._gru.a0, 0)
        h01s = T.unbroadcast(self._gru.h01, 0)
        h02s = T.unbroadcast(self._gru.h02, 0)
        
        rval, _ = theano.scan(
            fn=self._gru.step,
            sequences=[shuffled_xs, projected_xs],
            outputs_info=[a0s, h01s, h02s]
        )
        
        hs = rval[2]
        
        return hs.reshape((-1, self._gru.num_units))


class GRUStepLayer(L.MergeLayer):
    def __init__(self, incomings, gru_layer, name=None):
        super(GRUStepLayer, self).__init__(incomings, name)
        self._gru = gru_layer

    def get_params(self, **tags):
        return self._gru.get_params(**tags)

    def get_output_shape_for(self, input_shapes):
        n_batches = input_shapes[0][0]
        return n_batches, self._gru.num_units

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        
        uprev, x, aprev, hprev, h2prev = inputs
        
        # x [batch, channel, height, width]
        
        x = x.reshape(self._gru.step_shape)
        
        shuffled_x = x.dimshuffle(0, 2, 1)
        projected_x = T.dot(shuffled_x, self._gru.W_xa) + self._gru.b_p
        
        
        uprev = self._gru.W_ua[uprev]
        projected_x = projected_x + uprev[:,None,:]
        
        rval = self._gru.step(shuffled_x, projected_x, aprev, hprev, h2prev)
        
        self._gru.attention = rval[0]
        self._gru.hidden1 = rval[1]
        
        return rval[2]

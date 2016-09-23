import scipy
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
from lasagne.regularization import regularize_network_params, l2
import cPickle
from layers import build_cnn, GRULayer


MIXFRAC = 0.3


class BaselineLearner:
    """
    https://github.com/joschu/modular_rl/blob/master/modular_rl/core.py
    """

    def __init__(self, config):
        
        width = config['width']
        height = config['height']
        channels = config['channels']
        history = config['history']
        gru_units = config['gru_units']
        att_units = config['att_units']
        
        l_action = L.InputLayer((None,))
        
        l_input, l_cnn = build_cnn(config)
        
        l_gru = GRULayer(
            [l_action, l_cnn],
            num_steps=history,
            num_units=gru_units,
            att_units=att_units
        )
        
        l_gru_path = l_gru.get_path_layer([l_action, l_cnn])
        
        l_out_path = L.DenseLayer(
            l_gru_path,
            num_units=1,
            nonlinearity=None
        )

        l_out_batch = L.DenseLayer(
            l_gru,
            num_units=l_out_path.num_units,
            nonlinearity=l_out_path.nonlinearity,
            W=l_out_path.W,
            b=l_out_path.b
        )

        num_params_all = L.count_params(l_out_batch, trainable=True)
        params_all = L.get_all_params(l_out_batch, trainable=True) 
        shapes_all = [p.get_value(borrow=True).shape for p in params_all]
        
        num_params_cnn = L.count_params(l_cnn, trainable=True)
        params_cnn = L.get_all_params(l_cnn, trainable=True) 
        shapes_cnn = [p.get_value(borrow=True).shape for p in params_cnn]
        
        num_params_gru = num_params_all - num_params_cnn
        params_gru = [p for p in params_all if p not in params_cnn]
        shapes_gru = [p.get_value(borrow=True).shape for p in params_gru]

        print 'Number of baseline parameters: {} > {}({}) = {}({}) + {}({})'.format(
            L.count_params(l_out_path), num_params_all, len(params_all),
            num_params_cnn, len(params_cnn), num_params_gru, len(params_gru)
        )

        self.cnn = l_cnn
        self.gru = l_gru
        self.gru_path = l_out_path
        self.gru_batch = l_out_batch
        
        self.history = history
        self.batch_history_shape = (-1, history, channels, height, width)
        self.batch_flatten_shape = (-1, channels, height, width)

        action = T.ivector('action')
        state = T.tensor4('state')
        reward = T.fvector('reward')
        
        self.method = 'Adam'
        
        #l2_penalty = regularize_network_params(l_out_batch, l2)

        # Step 1.
        output = L.get_output(l_out_path,
                              {l_action:action, l_input:state},
                              deterministic=True).flatten()        
        print 'Compile baseline deterministic path output'
        self._output = theano.function([action, state], output)
        
        # Step 2. Here we can update parameters of batch_norm layers
        output = L.get_output(l_out_batch,
                              {l_action:action, l_input:state},
                              deterministic=False).flatten()
        loss = T.mean(T.square((output - reward)))# + 1e-4 * l2_penalty
        
        if self.method == 'Adam':
            
            # Step 3.a Adam updates
            
            grads = T.grad(loss, params_all)            
            #grads = [T.clip(g,-3,3) for g in grads]
            _, norm_before = lasagne.updates.total_norm_constraint(grads, 10, return_norm=True)
            
            updates = lasagne.updates.adam(grads, params_all, .0001)
            print 'Compile baseline Adam updates'
            self._compute_updates = theano.function(
                [action, state, reward],
                [loss, norm_before],
                updates=updates
            )  
        
        elif self.method == 'BFGS':
            
            print 'Compile baseline loss'
            self._compute_loss = theano.function([action, state, reward], loss)
            
            # Step 3.b Second order optimization
            # http://arxiv.org/pdf/1503.05671v6.pdf
            # QUOTE: the curvature matrix must remain fixed while CG iterates.
            output = L.get_output(l_out_batch,
                                  {l_action:action, l_input:state},
                                  deterministic=True).flatten()
            loss = T.mean(T.square((output - reward)))# + 1e-4 * l2_penalty
            grads = T.grad(loss, params_all)
            grads_flat = T.concatenate([g.flatten() for g in grads])
            print 'Compile baseline loss and flatten gradient'
            self._compute_loss_gradient = theano.function([action, state, reward], [loss, grads_flat])
            
            # Flatten parameters set/get
            theta = T.fvector(name="theta")
            offset = 0
            updates = []
            for p in params_all:
                shape = p.get_value().shape
                size = np.prod(shape)
                updates.append((p, theta[offset:offset+size].reshape(shape)))
                offset += size
            print 'Compile baseline set/get flatten params'
            self._set_params = theano.function([theta],[], updates=updates)
            self._get_params = theano.function([], T.concatenate([p.flatten() for p in params_all]))
        
        grads_norm = [T.sqrt(T.sum(g**2)) for g in grads]
        print 'Compile baseline gradient norm'
        self._compute_gradient_norm = theano.function([action, state, reward], grads_norm)

    def copy_gru(self, shared_layers):
        shared_weights = L.get_all_param_values(shared_layers)
        L.set_all_param_values(self.gru, shared_weights)
    
    def backup(self, name='baseline_best.pkl'):
        with open(name, 'wb') as file:
            cPickle.dump(L.get_all_param_values(self.gru_path), file)
            
    def restore(self, name='baseline_best.pkl'):
        with open(name, 'rb') as file:
            L.set_all_param_values(self.gru_path, cPickle.load(file))
            
    def train(self, A, V, X, Y):
        
        Y_mixfrac = Y*MIXFRAC + V*(1 - MIXFRAC)
        
        if self.method == 'BFGS':
        
            loss = self._compute_loss(A, X, Y_mixfrac)
            norm = 0

            def lossandgrad(th):
                self._set_params(th.astype(np.float32))
                l,g = self._compute_loss_gradient(A, X, Y_mixfrac)
                return (l,g.astype('float64'))

            thprev = self._get_params()
            thnext, _, _ = scipy.optimize.fmin_l_bfgs_b(lossandgrad, thprev, maxiter=30)

            thnext = 0.1 * thprev + 0.9 * thnext.astype(np.float32)

            self._set_params(thnext)
        
        elif self.method == 'Adam':
            
            loss = 0
            norm = 0
            updates = 0
            minibatch = 16

            A = A.reshape((-1, self.history))
            X = X.reshape(self.batch_history_shape)
            Y_mixfrac = Y_mixfrac.reshape((-1, self.history))

            N = X.shape[0]
            B = N // minibatch
            for _ in range(3):
                I = np.random.permutation(N)
                A = A[I]
                X = X[I]
                Y_mixfrac = Y_mixfrac[I]
                for b in range(B):
                    Ab = A[b*minibatch:(b+1)*minibatch].flatten()
                    Xb = X[b*minibatch:(b+1)*minibatch].reshape(self.batch_flatten_shape)
                    Yb = Y_mixfrac[b*minibatch:(b+1)*minibatch].flatten()
                    loss_batch, norm_batch = self._compute_updates(Ab, Xb, Yb)
                    if np.isnan(loss_batch):
                        raise TypeError("Loss function return NaN!")
                    loss += loss_batch
                    norm += norm_batch
                    updates += 1
            loss = float(loss) / updates
            norm = float(norm) / updates                        
        
        #print 'Ystdev {:.5f} / {:.5f} / {:.5f}'.format(V.std(), self._output(X).std(), Y.std())
        
        return (loss, norm)
    
    def debug(self, A, V, X, Y):
        Y_mixfrac = Y*MIXFRAC + V*(1 - MIXFRAC)
        W = L.get_all_param_values(self.gru_batch, trainable=True)
        G = self._compute_gradient_norm(A, X, Y_mixfrac)
        print '\nBaseline:'
        for w,g in zip(W, G):
            print "{:10s} \t {:8.5f} \t {:8.5f}".format(w.shape, np.sqrt((w**2).sum()), float(g))
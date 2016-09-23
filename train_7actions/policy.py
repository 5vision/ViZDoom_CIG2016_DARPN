import time
import numpy as np
import theano
import theano.tensor as T
import lasagne
import lasagne.layers as L
import cPickle
from layers import build_cnn, build_deconv, GRULayer
from theano_optimize.minresQLP import minresQLP


TINY = 1e-8
DAMPING = 0.01 # Add multiple of the identity to Fisher matrix

#MAX_KL = 0.0002 # KL divergence between old and new policy (averaged over state-space)
MAX_KL = 0.0001 # only for track2


class PolicyBase:
    
    def __init__(self, config, use_noise=False):
        
        self.width = config['width']
        self.height = config['height']
        self.channels = config['channels']
        self.actions = config['actions']
        self.history = config['history']
        
        gru_units = config['gru_units']
        att_units = config['att_units']
        
        l_action = L.InputLayer((None,))

        l_input, l_cnn = build_cnn(config, use_noise)
        
        l_gru = GRULayer(
            [l_action, l_cnn],
            num_steps=self.history,
            num_units=gru_units,
            att_units=att_units
        )
        
        l_attention = L.InputLayer((None, l_gru.num_pixel))
        l_hidden1 = L.InputLayer((None, gru_units))
        l_hidden2 = L.InputLayer((None, gru_units))
        
        l_gru_step = l_gru.get_step_layer([l_action, l_cnn, l_attention, l_hidden1, l_hidden2])
        
        l_out_step = L.DenseLayer(
            l_gru_step,
            num_units=self.actions,
            nonlinearity=lasagne.nonlinearities.softmax
        )

        l_out_batch = L.DenseLayer(
            l_gru,
            num_units=l_out_step.num_units,
            nonlinearity=l_out_step.nonlinearity,
            W=l_out_step.W,
            b=l_out_step.b
        )
        
        self.l_attention = l_attention
        self.l_action = l_action
        self.l_input = l_input
        
        self.num_params_all = L.count_params(l_out_batch, trainable=True)
        self.params_all = L.get_all_params(l_out_batch, trainable=True) 
        #shapes_all = [p.get_value(borrow=True).shape for p in params_all]
        
        self.num_params_cnn = L.count_params(l_cnn, trainable=True)
        self.params_cnn = L.get_all_params(l_cnn, trainable=True) 
        #shapes_cnn = [p.get_value(borrow=True).shape for p in params_cnn]
        
        self.num_params_gru = self.num_params_all - self.num_params_cnn
        self.params_gru = [p for p in self.params_all if p not in self.params_cnn]
        #shapes_gru = [p.get_value(borrow=True).shape for p in params_gru]

        print 'Number of policy parameters: {} > {}({}) = {}({}) + {}({})'.format(
            L.count_params(l_out_step), self.num_params_all, len(self.params_all),
            self.num_params_cnn, len(self.params_cnn), self.num_params_gru, len(self.params_gru)
        )

        self.cnn = l_cnn
        self.gru = l_gru
        self.gru_step = l_out_step
        self.gru_batch = l_out_batch
        
        self.batch_history_shape = (-1, self.history, self.channels, self.height, self.width)
        self.batch_flatten_shape = (-1, self.channels, self.height, self.width)

        self.t_action = T.ivector('action')
        self.t_state = T.tensor4('state')
        self.t_attention = T.matrix('attention')
        self.t_hidden1 = T.matrix('hidden1')
        self.t_hidden2 = T.matrix('hidden2')
        
        step_hidden2, step_output = L.get_output(
            [l_gru_step, l_out_step],
            {
                l_action:self.t_action,
                l_input:self.t_state,
                l_attention:self.t_attention,
                l_hidden1:self.t_hidden1,
                l_hidden2:self.t_hidden2
            },
            deterministic=True
        )
        step_hidden1 = l_gru.hidden1
        step_attention = l_gru.attention
        
        print 'Compile policy one step output'
        self._output_step = theano.function(
            [self.t_action, self.t_state, self.t_attention, self.t_hidden1, self.t_hidden2],
            [step_attention, step_hidden1, step_hidden2, step_output]
        )

    def backup(self, name='policy_best.pkl'):
        with open(name, 'wb') as file:
            cPickle.dump(L.get_all_param_values(self.gru_step), file)

    def restore(self, name='policy_best.pkl'):
        with open(name, 'rb') as file:
            L.set_all_param_values(self.gru_step, cPickle.load(file))

        
class PolicyTest(PolicyBase):

    def __init__(self, restore_file, config):
        
        PolicyBase.__init__(self, config, False)
        
        l_out_step_deconv = build_deconv(self.l_attention)
        step_deconv_attention = L.get_output(l_out_step_deconv, self.t_attention)
        self._deconv_attention = theano.function([self.t_attention], step_deconv_attention)
        
        self.restore(restore_file)
        
        self.reset_state()

    def output(self, action, state):
        action = np.array([action], np.int32)
        self.attention, self.hidden1, self.hidden2, p = self._output_step(
            action, state, self.attention, self.hidden1, self.hidden2
        )
        return p
    
    def reset_state(self):
        self.attention = self.gru.a0.get_value()
        self.hidden1 = self.gru.h01.get_value()
        self.hidden2 = self.gru.h02.get_value()
        
        
class PolicyPretrain(PolicyBase):

    def __init__(self, config):
        
        PolicyBase.__init__(self, config, True)
        
        #W = L.get_all_param_values(self.gru_batch, trainable=True)
        #print '\nPolicy:'
        #for w in W:
        #    w_norm = float(np.sqrt((w**2).sum()))
        #    print "{:10s} \t {:8.5f}".format(w.shape, w_norm)

        lr = T.scalar()
        vl = T.scalar()
        
        # Validate
        new_output_d = L.get_output(self.gru_batch,
                                    {self.l_action:self.t_action, self.l_input:self.t_state},
                                    deterministic=True)
        loss = T.mean(lasagne.objectives.categorical_crossentropy(new_output_d, self.t_action))
        print 'Compile policy validation'
        self._compute_validate = theano.function([self.t_state, self.t_action], loss)
        # Pretrain
        new_output_s = L.get_output(self.gru_batch,
                                    {self.l_action:self.t_action, self.l_input:self.t_state},
                                    deterministic=False)
        loss = T.mean(lasagne.objectives.categorical_crossentropy(new_output_s, self.t_action))
        updates = lasagne.updates.adam(loss, self.params_all, lr, vl)
        print 'Compile policy Adam updates'
        self._compute_updates = theano.function([self.t_state, self.t_action, lr, vl], loss, updates=updates)
    
    def validate(self, X, Y):
        loss = self._compute_validate(X, Y)
        if np.isnan(loss):
            raise TypeError("Loss function return NaN!")
        return float(loss)

    def pretrain(self, X, Y, lr=0.001, vl=0.9, minibatch=16):   
        updates = 0
        loss = 0
        
        X = X.reshape(self.batch_history_shape)
        Y = Y.reshape((-1, self.history))
        
        N = X.shape[0]
        B = N // minibatch
        assert B > 0
        
        for _ in xrange(2):
            I = np.random.permutation(N)
            X = X[I]
            Y = Y[I]
            for b in xrange(B):
                Xb = X[b*minibatch:(b+1)*minibatch].reshape(self.batch_flatten_shape)
                Yb = Y[b*minibatch:(b+1)*minibatch].flatten()
                loss += self._compute_updates(Xb, Yb, lr, vl)
                if np.isnan(loss):
                    raise TypeError("Loss function return NaN!")
                updates += 1
        return updates, loss
            
        
class PolicyLearner(PolicyBase):
    """
    https://github.com/rllab/rllab/blob/master/rllab/optimizers/conjugate_gradient_optimizer.py
    https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py
    https://github.com/rllab/rllab/blob/master/rllab/core/network.py
    """

    def __init__(self, config):
        
        PolicyBase.__init__(self, config, config['rl_noise'])
            
        self.n_updates = 0
        self.batch_size = 210 #config['batch_size'] * 2
        self.batch_limit = self.batch_size * 10
        
        # TODO: check version with batch statistics updates right here.
        # It will decrease KL divergence for the first run in current iteration.
        new_output_d = L.get_output(
            self.gru_batch,
            {self.l_action:self.t_action, self.l_input:self.t_state},
            #batch_norm_use_averages=True,
            #batch_norm_update_averages=False,
            deterministic=True
        )        
        print 'Compile policy batch output'
        self._output_batch = theano.function([self.t_action, self.t_state], new_output_d)
        
        
        ############################################################
        # Step 1. Compile function for computing eucledian gradients
        ############################################################            
        
        self.M_A = None
        self.M_S = None
        self.velocity = np.zeros(self.num_params_all, np.float32)
        
        self._s_np = np.zeros((1,self.channels,self.height,self.width), dtype=theano.config.floatX)
        self._p_np = np.zeros((1,self.actions), dtype=theano.config.floatX)
        self._a_np = np.zeros((1,), dtype=np.int32)
        self._r_np = np.zeros((1,), dtype=theano.config.floatX)
        
        self._s = theano.shared(self._s_np)
        self._p = theano.shared(self._p_np)
        self._a = theano.shared(self._a_np)
        self._r = theano.shared(self._r_np)
        
        indexes = T.arange(self._s.shape[0])
        
        # deterministic output for objectives
        
        new_output_d = L.get_output(self.gru_batch,
                                    {self.l_action:self._a, self.l_input:self._s},
                                    deterministic=True)
        new_output_log_d = T.log(new_output_d + TINY)
        
        likelihood_ratio = new_output_d[indexes, self._a] / self._p[indexes, self._a]
        surr_d = -T.mean(likelihood_ratio * self._r)
        kl = T.mean(T.sum(self._p * (T.log(self._p + TINY) - new_output_log_d), axis=1))
        entropy = T.mean(-(new_output_d * new_output_log_d).sum(axis=1))
        
        print 'Compile policy objectives'
        self._compute_objectives = theano.function([], [surr_d, kl, entropy])
        
        # stochastic output for gradients
        
        new_output_s = L.get_output(
            self.gru_batch,
            {self.l_action:self._a, self.l_input:self._s},
            #batch_norm_use_averages=True,
            #batch_norm_update_averages=False,
            deterministic=True # only for track2 else False
        )
        
        likelihood_ratio = new_output_s[indexes, self._a] / self._p[indexes, self._a]
        
        surr_s = -T.mean(likelihood_ratio * self._r)
        
        #surr_s += 0.01 * T.mean((new_output_s * T.log(new_output_s + TINY)).sum(axis=1))
        
        updates = []
        
        gs_all = T.concatenate([g.flatten() for g in T.grad(surr_s, self.params_all)])
        _gs_all = theano.shared(np.zeros(self.num_params_all, dtype=theano.config.floatX))
        updates.append([_gs_all, gs_all])
        
        gs_norm_all = T.sqrt(T.sum(gs_all ** 2))
        _gs_norm_all = theano.shared(np.float32(0))
        updates.append([_gs_norm_all, gs_norm_all])
        
        if config['rl_split']:
            
            gs_cnn = gs_all[:self.num_params_cnn]
            _gs_cnn = theano.shared(np.zeros(self.num_params_cnn, dtype=theano.config.floatX))
            updates.append([_gs_cnn, gs_cnn])
            
            gs_norm_cnn = T.sqrt(T.sum(gs_cnn ** 2))
            _gs_norm_cnn = theano.shared(np.float32(0))
            updates.append([_gs_norm_cnn, gs_norm_cnn])

            gs_gru = gs_all[self.num_params_cnn:]
            _gs_gru = theano.shared(np.zeros(self.num_params_gru, dtype=theano.config.floatX))
            updates.append([_gs_gru, gs_gru])
            
            gs_norm_gru = T.sqrt(T.sum(gs_gru ** 2))
            _gs_norm_gru = theano.shared(np.float32(0))
            updates.append([_gs_norm_gru, gs_norm_gru])
        
        start_time = time.time()
        print 'Compile policy eucledian gradients',
        self._compute_eucledian_gradients = theano.function([], [], updates=updates)
        print time.time() - start_time
        
        
        #############################################################
        # Step 2. Compile function for computing riemannian gradients
        #############################################################
        
        self._a2_np = np.zeros((1,), dtype=np.int32)
        self._a2 = theano.shared(self._a2_np)
        
        self._s2_np = np.zeros((1,self.channels,self.height,self.width), dtype=theano.config.floatX)
        self._s2 = theano.shared(self._s2_np)
        
        # http://arxiv.org/pdf/1503.05671v6.pdf
        # QUOTE: the curvature matrix must remain fixed while CG iterates.
        
        new_output = L.get_output(
            self.gru_batch,
            {self.l_action:self._a2, self.l_input:self._s2},
            #batch_norm_use_averages=True,
            #batch_norm_update_averages=False,
            deterministic=True
        )
        
        def solve_Ax_g(num_params, params, grads, grads_norm):

            def compute_Ax(x):
                
                # There are three ways to compute the Fisher-vector product:
                
                # 1. https://github.com/joschu/modular_rl/blob/master/modular_rl/trpo.py#L54
                # Use theano.gradient.disconnected_grad and call theano.tensor.grad() twice.
                # WARNING: In our case (with the attention mechanism) it is extremly slow.
                
                # 2. http://deeplearning.net/software/theano/tutorial/gradients.html#hessian-times-a-vector
                # Use only theano.tensor.Rop, but you will need to calculate the fixed_output outside
                # of the compiled function, because disconnected_grad will not work with Rop.
                
                # 3. https://github.com/pascanur/natgrad/blob/master/model_convMNIST_standard.py
                # Rop devided by output because a metric F is based on gradient of log(output).
                # Here we also split the vector of parameters. Not checked, but it may be
                # faster then supply few vectors to minresQLP.
                
                xs = []
                offset = 0
                for p in params:
                    shape = p.get_value().shape
                    size = np.prod(shape)
                    xs.append(x[offset:offset+size].reshape(shape))
                    offset += size
                    
                jvp = T.Rop(new_output, params, xs) / (new_output * self.batch_size * self.history + TINY)
                fvp = T.Lop(new_output, params, jvp)
                fvp = T.concatenate([g.flatten() for g in fvp])
                
                return [fvp], {}

            rvals = minresQLP(compute_Ax,
                              grads / grads_norm,
                              num_params,
                              damp=DAMPING,
                              rtol=1e-10,
                              maxit=40,
                              TranCond=1)

            flag = T.cast(rvals[1], 'int32')
            residual = rvals[3]
            Acond = rvals[5]

            x = rvals[0] * grads_norm
            Ax = compute_Ax(x)[0][0] + DAMPING * x
            xAx = x.dot(Ax.T)

            lm = T.sqrt(2 * MAX_KL / xAx)
            rs = lm * x
            
            return rs, lm, flag, residual, Acond
        
        
        if config['rl_split']:        
            rs_cnn, lm, flag, residual, Acond = solve_Ax_g(self.num_params_cnn, self.params_cnn, _gs_cnn, _gs_norm_cnn)
            rs_gru, lm, flag, residual, Acond = solve_Ax_g(self.num_params_gru, self.params_gru, _gs_gru, _gs_norm_gru)
            rs = T.concatenate([rs_cnn, rs_gru])
        else:
            rs, lm, flag, residual, Acond = solve_Ax_g(self.num_params_all, self.params_all, _gs_all, _gs_norm_all)
        
        slope = _gs_all.dot(rs.T)
        
        start_time = time.time()
        print 'Compile policy riemannian gradients', 
        self._compute_natural_gradients = theano.function([], [rs, slope, lm, flag, residual, Acond])
        print time.time() - start_time
        
        # Flatten parameters set/get
        theta = T.fvector(name="theta")
        offset = 0
        updates = []
        for p in self.params_all:
            shape = p.get_value().shape
            size = np.prod(shape)
            updates.append((p, theta[offset:offset+size].reshape(shape)))
            offset += size
        print 'Compile policy set/get flatten params'
        self._set_params = theano.function([theta],[], updates=updates)
        self._get_params = theano.function([], T.concatenate([p.flatten() for p in self.params_all]))

    def linesearch(self, fullstep, expected_improve_rate=None):
        
        momentum = 0.1 # only for track2 else 0.5
        alphas = 0.5**np.arange(3) / (20 + self.n_updates / 200.0)
        beta = min(1 - 1 / (self.n_updates / 2500.0 + 10 / 7.0), 0.9)
        
        x = self._get_params()

        momentum_velocity = momentum * self.velocity

        loss_before, kl_before, entropy_before = self._compute_objectives()
        
        for i, alpha in enumerate(alphas):

            #xnew = x - alpha * fullstep

            # https://github.com/lisa-lab/pylearn2/pull/136#issuecomment-10381617

            # regular momentum:
            velocitynew = momentum_velocity - alpha * fullstep
            xnew = x + velocitynew

            # alternate formulation for Nesterov momentum:
            #velocitynew = momentum_velocity - alpha * fullstep
            #xnew = x + momentum * velocitynew - alpha * fullstep
            
            xavg = beta * x + (1 - beta) * xnew

            self._set_params(xavg)
            
            loss, kl, entropy = self._compute_objectives()
            
            actual_improve = loss_before - loss

            expected_improve = expected_improve_rate * alpha + TINY
            
            ratio = actual_improve / expected_improve

            if actual_improve >= 0 and kl <= MAX_KL:
                self.n_updates += 1
                self.velocity = velocitynew
                return loss, kl, entropy, ratio, i

        self._set_params(x)
            
        print "WARNING: linesearch  {:.3f}/{:.3f}  {:.4f}/{:.4f}  {:.6f}/{:.6f}={:.3f}".format(
            float(loss_before), float(loss),
            float(kl_before), float(kl),
            float(actual_improve), float(expected_improve), float(ratio)
        )
        
        return loss, kl, entropy, ratio, i

    def train(self, S, P, A, R):
        
        P[P < TINY] = TINY
        
        # Step 1. Calculate the gradient for labeled data
        
        self._s.set_value(S, borrow=True)
        self._p.set_value(P, borrow=True)
        self._a.set_value(A, borrow=True)
        self._r.set_value(R, borrow=True)
        
        self._compute_eucledian_gradients()
        
        # Free memory on device
        self._s.set_value(self._s_np, borrow=True)
        self._p.set_value(self._p_np, borrow=True)
        self._a.set_value(self._a_np, borrow=True)
        self._r.set_value(self._r_np, borrow=True)
        
        # Step 2. Accumulate unlabled data for the Fisher matrix estimation
        
        M_A = A.reshape((-1, self.history))
        M_S = S.reshape(self.batch_history_shape)
        
        if self.M_S is None:
            self.M_A = M_A
            self.M_S = M_S
        else:
            if self.M_S.shape[0] > self.batch_limit:
                self.M_A = self.M_A[-self.batch_limit:]
                self.M_S = self.M_S[-self.batch_limit:]
            self.M_A = np.concatenate([self.M_A,M_A]).astype(np.int32)
            self.M_S = np.concatenate([self.M_S,M_S]).astype(np.float32)
        
        I = np.random.permutation(self.M_S.shape[0])
        A2 = self.M_A[I[:self.batch_size]].flatten()
        S2 = self.M_S[I[:self.batch_size]].reshape(self.batch_flatten_shape)
                
        self._a2.set_value(A2, borrow=True)
        self._s2.set_value(S2, borrow=True)
        
        rs, slope, lm, flag, residual, Acond = self._compute_natural_gradients()
        
        rs_norm = float(np.sqrt((rs**2).sum()))
        
        # Free memory on device
        self._a2.set_value(self._a2_np, borrow=True)
        self._s2.set_value(self._s2_np, borrow=True)
        
        if flag != 8:
            print 'WARNING: minresQLP', rs, slope, lm, flag, residual, Acond
            
        # Step 3. Backtracking linesearch, where expected_improve_rate is
        # the slope dy/dx at the initial point
        
        # Load data back for linesearch
        self._s.set_value(S, borrow=True)
        self._p.set_value(P, borrow=True)
        self._a.set_value(A, borrow=True)
        self._r.set_value(R, borrow=True)

        loss, kl, entropy, ratio, bt = self.linesearch(rs, slope)
        
        # Free memory on device
        self._s.set_value(self._s_np, borrow=True)
        self._p.set_value(self._p_np, borrow=True)
        self._a.set_value(self._a_np, borrow=True)
        self._r.set_value(self._r_np, borrow=True)
        
        return (loss, lm, kl, entropy, residual, Acond, ratio, bt, rs_norm)
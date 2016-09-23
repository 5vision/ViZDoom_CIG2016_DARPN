#import cplex
import numpy as np
import cPickle


ammo_weights = [
    2,   # 1. Fists
    50,  # 2. Pistol
    50,  # 3. Shotgun
    150, # 4. Chaingun
    10,  # 5. Rocket launcher
    300, # 6. Plasma gun
    300, # 7. BFG9000
]


def norm_reward(var_prev, var_next):
    var_prev = max(0, var_prev)
    var_next = max(0, var_next)
    var_sum = var_next + var_prev
    if var_sum == 0:
        return 0
    return float(var_next - var_prev) / var_sum
    


class ApprenticeshipLearner:
    """The apprenticeship learner.

    http://mlpy.readthedocs.io/en/latest/_modules/mlpy/learners/offline/irl.html

    The apprenticeship learner is an inverse reinforcement learner, a method introduced
    by Abbeel and Ng [1]_ which strives to imitate the demonstrations given by an expert.
    """

    def __init__(self):

        self._beta = 0.9
        self._gamma = 0.995
        self._rescale = False

        self._nfeatures = 5

        self._mu = []
        self._mu_E = []
        self.mu_mean = None
        self.mu_std = None
        
        self._t = 1.0
        self._thresh = np.finfo(np.float32).eps
        
        self._delta_last = 1.0

        """ Handcrafted weights
        """
        self._delta_thresh = 0.01
        self._weights = np.zeros(self._nfeatures)
        self._weights[0] = 1.0
        self._weights[1] = 0.5
        self._weights[2] = 0.5 # 0.3
        self._weights[3] = 0.5 # 0.7
        self._weights[4] = -0.005
        
        
    def backup(self, name='expert.pkl'):
        with open(name, 'wb') as file:
            cPickle.dump({'mean':self.mu_mean, 'std':self.mu_std, 'mu_E':self._mu_E}, file)

            
    def restore(self, name='expert.pkl'):
        with open(name, 'rb') as file:
            statistics = cPickle.load(file)
            self.mu_mean = statistics['mean']
            self.mu_std = statistics['std']
            self._mu_E = statistics['mu_E']
            print "Restored expert's statistics:\n{}\n{}\n{} = {}".format(
                self.mu_mean, self.mu_std, len(self._mu_E), self._mu_E[-1]
            )
    

    def get_state(self, var_prev=None, var_next=None):
        
        s = np.zeros(self._nfeatures)
        
        if var_prev is None or var_next is None:
            return s

        # Frags without suicide
        if var_next[0] > var_prev[0]:
            s[0] = min(var_next[0] - var_prev[0], 3)

        # Health with workaround for the telefrag
        if var_next[1] > -300:
            #s[1] = 2.0 * (var_next[1] - var_prev[1]) / (100 + var_prev[1])
            s[1] = norm_reward(var_prev[1], var_next[1])
        
        # Armor
        #s[2] = 2.0 * (var_next[2] - var_prev[2]) / (100 + var_prev[2])
        s[2] = norm_reward(var_prev[2], var_next[2])
        
        # Ammo
        if var_prev[4] < var_next[4]:
            s[3] = 0.5
        elif var_prev[4] > var_next[4]:
            s[3] = -0.25
        elif var_prev[4] > 1 and var_prev[4] < 8 and var_prev[3] != var_next[3]:
            #s[3] = 2.0 * (var_next[3] - var_prev[3]) / (ammo_weights[var_prev[4]-1] + var_prev[3] + var_next[3])
            #s[3] = norm_reward(var_prev[3], var_next[3])
            s[3] = np.clip(float(var_next[3] - var_prev[3]) / ammo_weights[var_prev[4]-1], -1, 1)
        
        # Living cost without Ammo
        if var_prev[3] == 0 and var_next[3] == 0:
            s[4] = 1
        
        return s
    

    def _estimate_mu(self, batch):
        """Estimate the experts/agent feature expectations.

        Calculate the empirical estimate for the experts/agent feature expectation mu
        from the observed trajectories.

        """

        mu = []
        
        for path in batch:
            
            t = 0
            mu_path = np.zeros(self._nfeatures)
            
            for i in range(len(path) - 1):
                
                var_prev = path[i][1]
                var_next = path[i+1][1]
                
                s = self.get_state(var_prev, var_next)
                mu_path += self._gamma ** t * s
                
                if var_prev[0] != var_next[0] or var_next[1] <= 0:
                    if t > 5 and var_next[1] > -300:
                        mu.append(mu_path)
                    t = 0
                    mu_path = np.zeros(self._nfeatures)
                else:
                    t += 1
                
        mu = np.vstack(mu)
        
        if self.mu_mean is None or self.mu_std is None:
            self.mu_mean = mu.mean(axis=0)
            self.mu_std = mu.std(axis=0)
            print "First expert's statistics:\n{}\n{}".format(self.mu_mean, self.mu_std)
            self.mu_std[self.mu_std == 0] = 1.0
        
        mu = (mu.mean(axis=0) - self.mu_mean) / self.mu_std

        if self._rescale:
            mu *= (1 - self._gamma)

        return mu


    def _compute_max_margin(self, mu):
        """ Inverse reinforcement learning step.

        Guesses the reward function being optimized by the expert; i.e. find the reward
        on which the expert does better by a 'margin' of `t`, then any of the policies
        found previously.

        Parameters
        ----------
        mu : array_like, shape (`n`, `nfeatures`)
            The set of feature expectations, where `n` is the number of iterations and
            `nfeatures` is the number of features.

        Returns
        -------
        t : float
            The margin.
        w : ndarray[float]
            The feature weights

        Notes
        -----
        Using the QP solver (CPLEX) solve the following equation:

        .. math::

            \\begin{aligned}
            & \\underset{t, w}{\\text{maximize}} & & t \\
            & \\text{subject to} & & w^T * mu_E > w^T * mu^j + t, j=0, \\ldots, n-1 \\
            & & & ||w||_2 <= 1.
            \\end{aligned}

        """
        try:
            n = len(mu)
            m = self._nfeatures

            cpx = cplex.Cplex()
            cpx.set_log_stream(None)
            #cpx.set_error_stream(None)
            #cpx.set_warning_stream(None)
            cpx.set_results_stream(None)
            cpx.objective.set_sense(cpx.objective.sense.maximize)

            obj = [1.0] + [0.0] * m
            lb = [0.0] + [-cplex.infinity] * m
            ub = [cplex.infinity] * (m + 1)
            names = ["t"]
            for i in range(m):
                names.append("w{0}".format(i))
            cpx.variables.add(obj=obj, lb=lb, ub=ub, names=names)

            expr = []
            for mu_E in self._mu_E:
                # add linear constraints:
                # w^T * mu_E >= w^T * mu^j + t, j=0,...,n-1
                #       => -t + w^T * (mu_E - mu^j) >= 0, j=0,...,n-1
                # populated by row
                for j in range(n):
                    row = [names, [-1.0] + (mu_E - mu[j]).tolist()]
                    expr.append(row)
                    
            k = len(self._mu_E) * n
            senses = "G" * k
            rhs = [0.0] * k
            cpx.linear_constraints.add(expr, senses, rhs)

            # add quadratic constraints:
            # w * w^T <= 1
            q = cplex.SparseTriple(ind1=names, ind2=names, val=[0.0] + [1.0] * m)
            cpx.quadratic_constraints.add(rhs=1.0, quad_expr=q, sense="L")

            cpx.solve()
            if not cpx.solution.get_status() == cpx.solution.status.optimal:
                raise Exception("No optimal solution found")

            t, w = cpx.solution.get_values(0), cpx.solution.get_values(1, m)
            w = np.array(w)
            return t, w

        except cplex.exceptions.CplexError as e:
            print e.message
            return None


    def add_expert(self, batch):
        self._mu_E.append(self._estimate_mu(batch))
        print "Expert {} = {}".format(len(self._mu_E), self._mu_E[-1])


    def get_reward(self, s):
        r = s.dot(self._weights.T)
        return r#np.clip(r,-1,1)


    def learn(self, batch):
        """Perform the inverse reinforcement learning algorithm.

        Parameters
        ----------
        batch : list
            The observations for the current policy.

        Returns
        -------
        bool :
            In the case of the algorithm having converged on the optimal policy,
            True is returned otherwise False. The algorithm is considered to have
            converged to the optimal policy if either the performance is within a
            certain threshold or if the maximum number of iterations has been reached.
        """
        
        if self._t <= self._thresh:
            return True

        # 2. Estimate mu
        self._mu.append(self._estimate_mu(batch))
        
        if len(self._mu) < 5:
            return False

        # 3. Compute maximum weights
        self._t, weights = self._compute_max_margin(self._mu)
        
        delta = abs(self._t - self._thresh)
        
        if delta >= self._delta_last:
            del self._mu[-1]
            return False
        
        self._delta_last = delta
        
        if delta < self._delta_thresh:
            self._weights = self._beta * self._weights + (1 - self._beta) * weights
            print "\nDelta error {0} weights = \n{1}\n{2}\n".format(delta, weights, self._weights)
            #print "mu = \n{}\nmu_E = \n{}".format(self._mu[-1], self._mu_E[-1])
            
        # 4. Check termination
        if self._t <= self._thresh:
            print "Converged to optimal solution"
            return True

        # 5. Compute optimal policy \pi^(i) using new weights ...

        return False

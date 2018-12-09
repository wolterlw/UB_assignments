import numpy as np

def Ber(p, size=1):
    return np.random.binomial(1,p, size=size)

class Users():
    def __init__(self, T, l, a, b, eps, tau, n, m, sizes, means):
        self.a = a
        self.b = b
        self.T = T
        self.eps = eps
        self.l = l
        self.cV = np.zeros((n,))
        self.cE = np.zeros((n,))
        self.tau = tau
        self.n = n
        assert n == sum(sizes), f"group sizes should add up to {n}"
        self.m = m
        self.sizes = sizes
        self.means = means
        self.est_last = np.zeros(n, dtype='float32')
        self.est_curr = np.zeros(n, dtype='float32')

    def get_true_avg(self):
    	return sum([s*m for s,m in zip(self.sizes, self.means)]) / self.n

    def groups(self):
        for i,s in enumerate(self.sizes):
            beg = sum(self.sizes[:i])
            group_idx = slice(beg, beg+s)
            yield i, group_idx

    def local_estimate(self):
        for i,idx in self.groups():
            self.est_curr[idx] = np.random.binomial(self.l,self.means[i], size=self.sizes[i])/self.l
        return self.est_curr
    
    def vote(self, t, debug=True):        
        log_T = int(np.floor(np.log(self.T)))
        diff = np.abs(self.local_estimate() - self.est_last)
        print(f"diff max: {diff.max():.4f}\ndiff mean: {diff.mean():.4f}")
        b_star = np.c_[[b*(diff > self.tau(b)) for b in range(-1, log_T+1)]].max(axis=0)
        
        if debug:
            print(f"b*={b_star.mean()}")
        VoteYes = (self.cV < self.eps/4) & np.logical_not(t % 2**(log_T - b_star))
        
        at = Ber(1 / (np.exp(self.a) + 1), size=self.n)
        at[VoteYes] = Ber(np.exp(self.a) / (np.exp(self.a) + 1), size=VoteYes.sum())
        return at
    
    def est(self):
        SendEstimate = self.cE < self.eps/4
        self.cE += self.b * SendEstimate
        
        p_t = Ber(1/(np.exp(self.b) + 1), size=self.n)
        p_t[SendEstimate] = Ber(
            (1 + self.local_estimate()*(np.exp(self.b)-1)) /\
            (np.exp(self.b) + 1), size=self.n
        )[SendEstimate]
        return p_t
    
    def fake_est(self, t):
        return Ber(1 / (np.exp(self.b) + 1), size=self.n)
    
    def update_p(self):
        self.est_last = self.est_curr.copy()
        
class Aggregator():
    def __init__(self, eps, delta, num_epochs, epoch_length, num_users, min_subgroup_size):
        self.eps = eps
        self.delta = delta
        self.T = num_epochs
        self.l = epoch_length
        self.n = num_users
        self.L = min_subgroup_size
        self.m = self.n // self.L
        tmp0 = np.log(12*self.m*self.T / delta)
        self.a = 4 * np.sqrt(2 * self.n * tmp0) /\
                 (self.L - 3/np.sqrt(2)*np.sqrt(self.n * tmp0))
        # vote noise level
        tmp1 = np.log(12*self.T / delta) / 2
        self.b = np.sqrt(2 * tmp1/self.n) /\
                 (np.log(self.T)*np.sqrt(tmp1 / self.l) - np.sqrt(tmp1 / self.n))
        # estimate noise level
        
        self.last_pub = -1
        self.p_last = -1
        
        right = (3/np.sqrt(2) + np.sqrt(32)/self.eps)*np.sqrt(self.n*np.log(12*self.m*self.T / self.delta))
        assert self.L > right, f"Assumption 4.2: {self.L} < {right}"
    
    def accuracy_guarantee(self):
        return 4*(np.floor(np.log(self.T)) + 2) * \
    		   np.sqrt(np.log(12*self.n*self.T / self.delta) / (2*self.l))
        
    def init_users(self):
        def tau(b):
            return 2 * (b + 1) * np.sqrt(np.log(10 * self.n * self.T / self.delta) / (2*self.l))
        return {'a': self.a, 'b': self.b, 'tau': tau}
    
    def epoch(self, t, users):
        ats = users.vote(t)
        
        print(np.mean(ats))
        
        val = (1/(np.exp(self.a) + 1) + np.sqrt(np.log(10*self.T/self.delta)/(2*self.n)))
        print(val)
        global_update = (
            np.mean(ats) > val
        )
        
        
        if global_update:
            print("Global update!")
            self.last_pub = t
            users.update_p()
            pts = users.est()
            self.p_last = np.mean( (pts*(np.exp(self.b)+1) - 1) / (np.exp(self.b) - 1) ) #not clear how this is supposed to be accurate
        else:
            pts = users.fake_est(t)
        return self.p_last

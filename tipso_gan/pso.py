import numpy as np
class AdaptivePSO:
    def __init__(self, swarm_size=80, w_min=0.4, w_max=0.9, c1=2.05, c2=2.05,
                 sigma_share=0.5, alpha_share=1.0, beta_randn=1e-3,
                 k_sigmoid=6.0, f_threshold=1e-4, delta_t=10, stagnation_T=25):
        self.N = swarm_size
        self.w_min = w_min; self.w_max = w_max
        self.c1 = c1; self.c2 = c2
        self.sigma_share = sigma_share; self.alpha_share = alpha_share
        self.beta_randn = beta_randn; self.k_sigmoid = k_sigmoid
        self.f_threshold = f_threshold; self.delta_t = delta_t
        self.stagnation_T = stagnation_T
        self.fbest_hist = []
    def _sigmoid_inertia(self, df):
        return self.w_min + (self.w_max - self.w_min) * (1.0 / (1.0 + np.exp(-self.k_sigmoid*(df - self.f_threshold))))
    def _fitness_sharing(self, fitness, X):
        dists = np.linalg.norm(X[:,None,:] - X[None,:,:], axis=-1)
        sh = np.maximum(0.0, 1.0 - (dists / (self.sigma_share + 1e-8))**self.alpha_share)
        denom = np.sum(sh, axis=1) + 1e-8
        return fitness / denom
    def optimize(self, dim, objective, max_iter=150):
        X = np.random.randn(self.N, dim) * 0.1
        V = np.zeros_like(X)
        pbest = X.copy()
        f_pbest = np.full((self.N,), np.inf)
        f_gbest = np.inf
        gbest = np.zeros((dim,))
        stagn = np.zeros((self.N,), dtype=int)
        for it in range(max_iter):
            f = np.array([objective(x) for x in X])
            f_shared = self._fitness_sharing(f, X)
            improved = f_shared < f_pbest
            pbest[improved] = X[improved]
            f_pbest[improved] = f_shared[improved]
            stagn[~improved] += 1
            stagn[improved] = 0
            idx = np.argmin(f_pbest)
            if f_pbest[idx] < f_gbest:
                f_gbest = f_pbest[idx]
                gbest = pbest[idx].copy()
            if len(self.fbest_hist) > self.delta_t:
                df = self.fbest_hist[-1] - self.fbest_hist[-1 - self.delta_t]
            else:
                df = 0.0
            self.fbest_hist.append(f_gbest)
            w = self._sigmoid_inertia(df)
            r1 = np.random.rand(self.N, dim)
            r2 = np.random.rand(self.N, dim)
            V = w*V + self.c1*r1*(pbest - X) + self.c2*r2*(gbest - X) + self.beta_randn*np.random.randn(self.N, dim)
            X = X + V
            reinit = stagn >= self.stagnation_T
            if np.any(reinit):
                X[reinit] = np.random.randn(np.sum(reinit), dim) * 0.1
                V[reinit] = 0.0
                f_pbest[reinit] = np.inf
                stagn[reinit] = 0
        return gbest, f_gbest

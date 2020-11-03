import numpy as np

# Helpers for the class
def integrate(f,x,u,dt):
    k1 = f(x,u)*dt
    k2 = f(x+k1/2.0,u)*dt
    k3 = f(x+k2/2.0,u)*dt
    k4 = f(x+k3,u)*dt
    return x + (k1+2.0*(k2+k3)+k4)/6.0

def simulate(f,x0,u_vec,dt,N):
    x = np.copy(x0)
    xtraj = np.zeros((len(x0),N))
    if len(u_vec.shape) == 1:
        u_vec = np.repeat(u_vec.reshape(-1,1),N,axis=1)
    for i in range(N):
        xtraj[:,i] = integrate(f,x,u_vec[:,i],dt)
        x = np.copy(xtraj[:,i])
    return xtraj

class MPPI():
    """
    This class implements a Model-Predictive Path Integral (MPPI)
    control scheme from "Information Theoretic MPC for Model-Based
    Reinforcement Learning" by Evangelos Theodorou (ICRA 2017).
    """
    def __init__(self, dynamics, objective_fun, x0, u_init=np.zeros(2), K=1000, N=100, dt = 0.1, lamb=1.0, noise_mu=0, noise_sigma=1):
        self.K = K  # total number of random trajectory samples
        self.dt = dt # timestep resolution
        self.N = N # total number timesteps to take
        self.lamb = lamb # temperature parameter
        self.cost_total = np.zeros(self.K) # total cost of each sampled trajectory
        self.cost_fun = objective_fun # objective function passed into class
        self.dynamics = dynamics # dynamics function passed into class
        self.x0 = x0 # initial condition
        self.noise_mu = noise_mu # control distribution mean
        self.noise_sigma = noise_sigma # control distribution covvariance
        if len(u_init.shape) == 1: # default control signal
            self.u_vec = np.repeat(u_init.reshape(-1,1),self.N,axis=1)
        else:
            self.u_vec = u_init

    def resample(self):
        # sample control noise vectors
        self.noise = np.random.multivariate_normal(mean=self.noise_mu, cov=self.noise_sigma, size=(self.K, self.N))

    def compute_sample_costs(self):
        # calculate the total costs corresponding to sampled trajectories
        for k in range(self.K):
            perturbed_u = self.u_vec + self.noise[k].T
            xtraj = simulate(self.dynamics,self.x0,perturbed_u,self.dt,self.N)
            self.cost_total[k] = self.cost_fun(xtraj,perturbed_u)

    def compute_control(self):
        # update control law based on sampled trajectories
        beta = np.min(self.cost_total)
        exp_form_cost = np.exp(-(1./self.lamb)*(self.cost_total-beta))
        eta = np.sum(exp_form_cost)
        omega = exp_form_cost/eta
        self.u_vec += np.array([np.sum(omega.reshape(-1,1)*self.noise[:, i],axis=0) for i in range(self.N)]).T        

    def step(self,x0,u_new=None):
        # do a full iteration of this process and return control vector
        self.cost_total[:] = 0
        self.x0 = np.copy(x0)
        self.resample()
        if u_new is None:
            self.compute_sample_costs()
        else:
            self.u_vec = np.copy(u_new)
            self.compute_sample_costs()
        self.compute_control()
        out_vec = np.copy(self.u_vec)
        self.u_vec = np.roll(self.u_vec, -1)
        return out_vec

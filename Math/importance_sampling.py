import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

class Pdf:
    def __call__(self, x):
        pass
    def sample(self,n):
        pass

class Norm(Pdf):
    def __init__(self,mu=0,sigma=1):
        self.mu = mu
        self.sigma = sigma
    def __call__(self,x):
        result=-0.5*(x-self.mu)**2/self.sigma**2
        print("p",result)
        return result
    def sample(self,n):
        return np.random.normal(self.mu,self.sigma,n)
class Uniform(Pdf):
    def __init__(self,low,high) -> None:
        self.low=low
        self.high=high
    def __call__(self,x):
        result=np.repeat(-np.log(self.high-self.low),len(x))
        print("q",result)
        return result
    def sample(self,n):
        return np.random.uniform(self.low,self.high,n)
class ImportanceSampler:
    def __init__(self,p_dist,q_dist) -> None:
        # pass
        self.p_dist=p_dist
        self.q_dist=q_dist
    def calc_weights(self,samples):
        return self.p_dist(samples)-self.q_dist(samples)
    def sample(self,n):
        samples=self.q_dist.sample(n)
        print(samples)
        weights=self.calc_weights(samples)
        norm_weights=weights-logsumexp(weights)
        print(norm_weights)
        return samples,norm_weights

if __name__=="__main__":
    # a=Norm(0,1)
    # a(1)
    # a.__call__(1)
    # pass
    N=5
    # Y=10
    target_p=Norm()
    imp_q=Uniform(-20,30)
    # imp_q(Y)
    sampler=ImportanceSampler(target_p,imp_q)
    baised_samples,logws=sampler.sample(N)
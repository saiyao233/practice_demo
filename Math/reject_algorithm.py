import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 1.2-x**4

def sample(f,xmin=0,xmax=1,ymax=1.2):
    while True:
        x=np.random.uniform(low=xmin,high=xmax)
        y=np.random.uniform(low=0,high=ymax)
        if y<f(x):
            return x
def batch_sample(f,num_samples,xmin=0,xmax=1,ymax=1.2,batch_size=1000):
    samples=[]
    while len(samples)<num_samples:
        xs=np.random.uniform(low=xmin,high=xmax,size=batch_size)
        ys=np.random.uniform(low=0,high=ymax,size=batch_size)
        samples.extend(xs[ys<f(xs)].tolist)
    return sample[:num_samples]
if __name__=="__main__":
    samples=[sample(f) for _ in range(10)]
    print(samples)
   
import numpy as np
# Reimann sum
# f: function
def cos(x):
    return np.cos(x)
def reimain_sum(f,a,b,N=1000):
    total_sum = 0
    for i in range(1,N+1):
        total_sum+=f(a+(b-a)/N*i)
    result=(b-a)/N*total_sum
    print(result)
    return result
# reimain_sum(cos,0,np.pi/2,1000)
# expected value
from scipy import stats
rv=stats.norm()
def expected(rv,N=10**6):
    result=sum([rv.rvs() for _ in range(N)])/N
    print(result)
    return result
def calc_int(f,a,b,N=10**6):
    delta=(b-a)/N
    height=sum(map(f,np.random.uniform(a,b,N)))
    result=delta*height
    print(result)
    return result

if __name__=="__main__":
    # reimain_sum(cos,0,np.pi/2,1000)
    # expected(rv,10**6)
    # s=np.random.normal(1,1,10)
    # print(s)
    # f=lambda x: x+1
    # result=map(f,[1,2])
    # print(list(result))
    calc_int(cos,0,np.pi/2)
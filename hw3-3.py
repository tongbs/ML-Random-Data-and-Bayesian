import random
import math
import numpy as np
from matplotlib import pyplot as plt

cond = True
def gaussian_data_generator(mean, variance):
    # Box-Muller transform
    U = random.random()
    V = random.random()
    global cond
    if cond:
        X = math.sqrt((-2)*math.log(U)) * (math.cos(2*math.pi*V))
    else:
        X = math.sqrt((-2)*math.log(U)) * (math.sin(2*math.pi*V))
    cond = not cond    
    return math.sqrt(variance)*X + mean

def polynomial_basis_data_generator(n,a,w):
    e = gaussian_data_generator(0,a)
    x = np.random.uniform(-1,1)   # uniformly distributed(-1<x<1)
    y = 0
    for i in range(n):
        y += w[i]*math.pow(x,i)
    y+=e
    print(f'Add data point ({x}, {y}):')
    return x ,y

def plotline(x, w, cov, Ground):
    # w: posterior mean, 
    up_y = []    # red line
    y = []       # black line
    down_y = []  # red line  
    for _x in x:
        pow_x = []
        print("545445454525",n)
        for j in range(n):
            pow_x.append(math.pow(_x,j))
        print(pow_x)
        pred_y = np.dot(pow_x, w)
        y.append(pred_y)
        if Ground == "Yes":
            up_y.append(pred_y+cov)
            down_y.append(pred_y-cov)
        else:
            X_array = np.array(pow_x).reshape((1,n))
            var_y = (1/a) + np.dot(np.dot(X_array,cov), tran_X_array)
            var_y = var_y[0][0]
            up_y.append(pred_y+var_y)
            down_y.append(pred_y-var_y)
    return y, up_y, down_y 

def visual(x, w, cov, data_x, data_y, Ground, title):
    y, up_y, down_y = plotline(x, w, cov, Ground)
    plt.title(title)
    plt.plot(x , y , "k")     # black
    plt.plot(x, up_y , "r")   # red
    plt.plot(x, down_y, "r")  # red

    if Ground != "Yes":
        plt.scatter(data_x, data_y, color="blue")
    plt.show()

if __name__ == "__main__":
    b = float(input("b = ")) # alpha
    n = int(input("n = ")) 
    a = float(input("a = ")) # beta = 1/a
    w = input("w = ")        # weight
    w = list(map(float,w.split(",")))
    
    iter_time = 0
    data = [[],[]]  # store point(x, y)

    # initial prior w ~ N(0, b^-1*I)
    prior_mean = np.zeros((n,1))      # m0
    prior_cov = (1/b)*np.identity(n)  # S0
    
    while True:
        iter_time+=1
        print(f'{iter_time}----------------------')
        x, y = polynomial_basis_data_generator(n, a, w)
        data[0].append(x)
        data[1].append(y)
        # --------------------------------------------------------------------------------------
        # y = t                                                                                - 
        # Φ design matrix                                                                      -
        # mN = mean of posterior = SN(inverse(S0)*m0 + beta*transpose(Φ)*t)                    -
        # SN = covariance matrix of posterior = inverse( inverse(S0) + beta*transpose(Φ)*Φ )   -
        # --------------------------------------------------------------------------------------
        beta = 1/a
        t = y
        tmp = []
        for i in range(n):
            tmp.append(math.pow(x,i))
        X_array = np.array(tmp).reshape((1,n))
        #print(X_array)
        inv_prior_cov = np.linalg.inv(prior_cov)
        tran_X_array = X_array.transpose()
        posterior_cov = np.linalg.inv( inv_prior_cov + beta*np.dot(tran_X_array, X_array))
        posterior_mean = np.dot(posterior_cov, ( np.dot(inv_prior_cov, prior_mean) + beta*tran_X_array*t ))
        print('Posterior mean: ')
        print(posterior_mean)
        print('Posterior variance:')
        print(posterior_cov)

        pred_y = np.dot(X_array, posterior_mean)
        print(pred_y)
        pred_y = pred_y[0][0]
        var_y = (1/beta) + np.dot(np.dot(X_array, posterior_cov), tran_X_array)
        var_y = var_y[0][0]
        print(f'predictive distribution ~ N({pred_y}, {var_y})')
        if iter_time > 3000:
            f_w_pred = posterior_mean.reshape(n)
            f_cov = posterior_cov
            break
        else:
            prior_mean = posterior_mean
            prior_cov = posterior_cov
            if iter_time == 10:
                w_10pred = posterior_mean.reshape(n)
                cov_10pred = posterior_cov
            elif iter_time == 50:
                w_50pred = posterior_mean.reshape(n)
                cov_50pred = posterior_cov
        
    x = np.linspace(-2,2,100)
    cov = a
    visual(x, w, cov, None, None, "Yes", "Ground truth")
    visual(x, f_w_pred, f_cov, data[0], data[1], "No", "Predict result")
    visual(x, w_10pred, cov_10pred, data[0][:10], data[1][:10], "No", "After 10 incomes")
    visual(x, w_50pred, cov_50pred, data[0][:50], data[1][:50], "No", "After 50 incomes")
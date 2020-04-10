import random
import math
import numpy as np

cond = False
def gaussian_data_generator(mean, variance):
    # Box-Muller transform
    U = random.random()
    V = random.random()
    global cond
    cond = not cond
    if cond:
        X = math.sqrt((-2)*math.log(U)) * (math.cos(2*math.pi*V))
    else:
        X = math.sqrt((-2)*math.log(U)) * (math.sin(2*math.pi*V))    
    return math.sqrt(variance)*X + mean

def sequential_estimator(mean, variance):
    old_mean = gaussian_data_generator(mean, variance)
    old_var = 0
    cnt = 1
    while True:
        new_data = gaussian_data_generator(mean, variance)
        new_mean = ((old_mean*cnt)+new_data) / (cnt+1)
        new_var = (((cnt-1)/cnt)*old_var) + (((new_data-new_mean)*(new_data-old_mean)) / (cnt+1))
        print(f'Add data point: {new_data}')
        print(f'Mean = {new_mean}    Variance = {new_var}')
        if(abs(new_mean-old_mean) < 0.001 and abs(new_var-old_var) < 0.001):
            break
        else:
            old_mean = new_mean
            old_var = new_var
            cnt+=1
if __name__ == "__main__":
    mean = float(input("Mean = "))
    variance = float(input("Variance = "))
    print(f'Data point source function: N({mean}, {variance})')
    sequential_estimator(mean, variance)
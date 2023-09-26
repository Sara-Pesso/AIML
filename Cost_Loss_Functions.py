def MSE(square_errors): #Mean Squared Error (also called quadratic cost/loss)
    return sum(square_errors)/len(square_errors)
#############
# Saturated neurons: activation function outputs that are very extreme-- causes learning to slow drastically

from math import log
def CrossEntropy_Clear(y, yhat):
    n = len(y)
    sum = 0
    for i in range(n):
        sum += y[i]*log(yhat[i]) + (1-y[i])*log(1-yhat[i])
    
    return -sum/n

def CrossEntropy(y, a):
    return -sum([y[i]*log(a[i])+ (1-y[i])*log(1-a[i]) for i in range(len(y))])/len(y)

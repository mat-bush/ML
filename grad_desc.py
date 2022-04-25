import numpy as np

def grad_descent(x,y,m1_guess,m0_guess,alpha0):
    # intitializes params
    m1 = m1_guess
    m0 = m0_guess
    alpha = alpha0
   

    

    # runs grad_desc algo

    for k in range(0,10000):

        dm1 = 0
        dm0 = 0
        for i in range(0,len(x)):
            
            dm1 = x[i] * (m1*x[i] + m0 - y[i]) + dm1
            dm0 = m1*x[i] + m0 - y[i] + dm0
        
        
        
        
        m1  = m1 - alpha * dm1/len(x)
        m0 = m0 - alpha * dm0/len(x)


    return [m1,m0]
import numpy as np

def grad_descent(x,y,m1_guess,m0_guess,alpha0,init_step):
    # intitializes params
    m1 = m1_guess
    m0 = m0_guess
    alpha = alpha0
    step1 = init_step
    step0 = init_step


    # runs grad_desc algo

    sse = 0

    for i in range(0,len(x)):
        sse = sse + np.float_power((m1*x[i] + m0 - y[i]),2)

    m1_new = m1 - step1
    m0_new = m0 - step0

    sse_new = 0
    for j in range(0,len(x)):
        sse_new = sse_new + np.float_power((m1_new*x[i] + m0_new - y[i]),2)

   
    


# function test
x =[2.4,3.67,3,4]
y = [.5,.7,.6,.8]

print(grad_descent(x,y,3,1,1,.1))
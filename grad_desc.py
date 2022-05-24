import random
import numpy as np
from com_classes.lin_al import mat_gen as mg




class grad_desc():
    def __init__(self):
        self.mat = mg.mat_gen()
        
    
    def sin_var_lin_reg(self,x,y,m1_guess,m0_guess,alpha0):
        # intitializes params
        m1 = m1_guess
        m0 = m0_guess
        alpha = alpha0
    
        if len(x) != len(y):
            print("Error: lenght 'x' does not equal lenght 'y'")
        else:  
        

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

    def multivar(self,X,Y):

        X = self.feature_scale(X)

        dm = np.zeros((np.shape(X)[1],1))

        M = self.mat.random(np.shape(X)[1],1) 
            
        J = np.dot(X,M) - Y
        for j in range(0,1000):
            for i in range(0,len(dm)):
                x_i = [X[:,i]]

                dm[i] = np.dot(x_i,J)

            M = M - dm

        return dm


    def feature_scale(self,X):
            
        # iterates thru columns
        for i in range(0,np.shape(X)[1]):
            col = X[:,i]
            mu = np.mean(col)

            for j in range(0,len(col)):
                X[j][i] = (X[j][i] - mu) / (max(col) - min(col))

    
        return X


mat = mg.mat_gen()

x = mat.random(3,2,float,1000,-1000)
m = mat.random(4,1)
y = mat.random(3,1)
# print(np.dot(x,m))
gd = grad_desc()

print(gd.multivar(x,y))

# print(np.zeros((12,3)))
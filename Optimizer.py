import math
import torch

class Optimizer():
    def __init__(self):
        pass
    
    def step(self):
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self, model_parameters, eta=1e-4, wd = False):
        super(SGD, self).__init__()
        """
            Model parameters to be optimized over (i.e. w, dw, b, db)
            lr the learning rate to be passed as argument, 1e-4 by default
        """
        self.parameters = model_parameters
        self.eta = eta
        self.wd = wd
        
    def step(self):
        """
            In param, we have p[0] = parameter, p[1] = dloss/dp
            i.e. p(t+1) = p(t) - eta * dl_dp
        """
        for p in self.parameters: 
            if not self.wd:
                p[0] = p[0] - self.eta * p[1]
            if self.wd:
                p[0] = p[0] - self.eta * p[1] - 2*self.wd*p[0]

#class SGD_momentum(Optimizer):
#    
#    def __init__(self,model_parameters,eta=1e-1,wd = False, gamma = 0.5):
#        self.parameters = model_parameters
#        self.eta = eta
#        self.gamma = gamma
#        self.wd = wd
#        #Initially, for t = 0, the momentum is zero as it has not moved yet.
#        #then at each step, "previous" will be updated.
#        self.previous_w = model_parameters.zero
#    
#    def step(self):
#        print(self.previous)
#        for par in self.parameters:
#            print(len(par[1]))
#            i=0
#            print(i)
#            if not self.wd:
#                difference = (self.eta*par[1]-self.previous[i])
#                par[0] = par[0] - self.eta*par[1]-gamma*self.previous[i]
#                #this saves the step of this move in memory to be re-used at the next step for momentum.
#                self.previous[i] = difference
#            if self.wd:
#                difference = (self.eta*par[1]-self.previous[i])
#                par[0] = par[0] - self.eta*par[1]-gamma*self.previous[i]-2*self.wd*par[0]
#                #this saves the step of this move in memory to be re-used at the next step for momentum.
#                self.previous[i] = difference
#            i+=1
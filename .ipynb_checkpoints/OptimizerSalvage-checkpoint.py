import math
import torch

class Optimizer():
    def __init__(self):
        pass
    
    def step(self):
        raise NotImplementedError
    
class SGD(Optimizer):
    def __init__(self, model_parameters, eta=1e-1):
        super(SGD, self).__init__()
        """
            Model parameters to be optimized over (i.e. w, dw, b, db)
            lr the learning rate to be passed as argument, 1e-1 by default
            No momentum
        """
        self.parameters = model_parameters
        self.eta = eta
        #print("HERE IS PRINT BEFORE UPDATE \n",self.parameters[0][0][0])
    def step(self):
        """
            In param, we have p[0] = parameter, p[1] = dloss/dp
            i.e. p(t+1) = p(t) - eta * dl_dp
        """
        for p in self.parameters: 
            #print("before update",p[0][0])
            p[0] = p[0] - self.eta * p[1]
            #print("Has this been modified properly",p[0][0])
        #print("SECOND PRINT HERE \n",self.parameters[0][0][0])
        #output = self.parameters
        #return output

#class SGD_momentum(Optimizer):
#    
#    def __init__(self,model_parameters,eta=1e-1,gamma = 0.5):
#        self.parameters = model_parameters
#        self.eta = eta
#        self.gamma = gamma
#        #Initially, for t = 0, the momentum is zero as it has not moved yet.
#        #then at each step, "previous" will be updated.
#        self.previous = torch.zeros(self.parameters[0].size())
#    
#    def step(self):
#        for par in self.parameters:
#            difference = (self.eta*par[1]-self.previous)
#            par[0] = par[0] - self.eta*par[1]-gamma*self.previous
#            
#            #this saves the step of this move in memory to be re-used at the next step for momentum.
#            self.previous = difference
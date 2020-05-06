class Optimizer():
    def __init__(self,model_parameters,*args):
        raise NotImplementedError
    
    def step(self):
        raise NotImplementedError
    
    
class SGD(Optimizer):
    def __init(self, model_parameters, eta= 1e-1):
        super(SGD, self).__init__()
        """
            Model parameters to be optimized over (i.e. w, dw, b, db)
            lr the learning rate to be passed as argument, 1e-1 by default
            No momentum
        """
        self.parameters = model_parameters
        self.eta = eta
        
    def step(self):
        """
            In param, we have p[0]= w, p[1] = dw
            p[2] = b, p[3] = db then we need to update par[0] = par[0]-eta*par[1],
            i.e. w(t+1) = w(t) - eta * dl_dw
        """
        for par in self.parameters : 
            #weights
            par[0] = par[0] - self.eta * par[1]
            #bias
            par[2] = par[2] - self.eta * par[3]
            
class SGD_momentum(Optimizer):
    
    def __init__(self,model_parameters,eta=1e-1,gamma = 0.5):
        self.parameters = model_parameters
        self.eta = eta
        self.gamma = gamma
        #Initially, for t = 0, the momentum is zero as it has not moved yet.
        #then at each step, "previous" will be updated.
        self.previous = torch.zeros(self.parameters[0].size())
    
    def step(self):
        for par in self.parameters:
            difference = (self.eta*par[1]-self.previous)
            par[0] = par[0] - self.eta*par[1]-gamma*self.previous
            
            #this saves the step of this move in memory to be re-used at the next step for momentum.
            self.previous = difference
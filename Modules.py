import math
import torch

class Module() :
    def __init__(self):
        pass
    
    def forward(self, *input):
        raise NotImplementedError
    
    def backward(self,*gradwrtoutput):
        raise NotImplementedError
    
    def param(self):
        pass
        
    def __call__(self, *input):
        return self.forward(*input)
    
class Linear(Module):
    """
        One fully connected layer.
        Input : input_dimension, output_dimension [int]
        Those corresponds to the numbers of nodes in a given input layer
        and the number of nodes in the output layer
        Weights and Bias are initialized using the Xavier initialization
        see course 5.5 slide 14 (What is the gain here??)
    """
    def __init__(self,in_dim,out_dim):
        super(Linear, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        std = math.sqrt(2./(in_dim+out_dim))
        self.w = torch.empty(in_dim, out_dim).normal_(0,std)
        self.b = torch.empty(out_dim).normal_(0,std)
        self.dw = torch.zeros(in_dim, out_dim)
        self.db = torch.zeros(out_dim)
        self.current = torch.empty(1,1)
    
    def param(self):
        return [[self.w, self.dw], [self.b, self.db]]
        
    def forward(self,x):
        """
            results of FW pass of layer, returns s = x*w + b
            using formula from Practical 03
        """
        self.current = x
        return x.mm(self.w)+self.b
   
    def backward(self, dl_ds):
        self.dw = self.current.t().mm(dl_ds)
        self.db = dl_ds.sum(0)
        return dl_ds.mm(self.w.t())
    
    def zero_grad(self):
        """
            Function to set grad to zero. Used in gradient step
        """
        self.dw.zero_()
        self.db.zero_()
    
    def step(self, eta=1e-4, wd=3e-5):
        self.w = self.w - eta * self.dw - 2*wd*self.w
        self.b = self.b - eta * self.db - 2*wd*self.b
        
class Sequential(Module):
    """
        An instance of the sequential class contains the Modules passed to it as arguments.
        For example, pass linear, relu to it, it will be stored within its attribute "members"
        the attribute memory is used to save the operations in the order in which they were done
        to allow backprop (?)
    """
    def __init__(self,*Modules):
        super(Sequential, self).__init__()
        self.members = Modules
        
    def param(self):
        parameters = []
        for module in self.members:
            if module.param() is not None:
                for p in module.param():
                    parameters.append(p)
        return parameters
    
    def forward(self, x):
        #MUST ADD THINGS TO MEMORY AFTER OPERATION.
        # x is the input to which the operations of each module must be applied in the right order.
        #Does this need to save the first entry ?
        #It should save the value at each step (ex what is x1, s1, sigma(s1), etc.)
        #this is done in the 
        #Should backprop also have another memory to save the backprop'd gradient and what else? Loss??
        
        for module in self.members:
            x = module(x)
        return x
    
    def backward(self, dl_dx):
        for module in reversed(self.members):
            dl_dx = module.backward(dl_dx)
    
    def zero_grad(self):
        #should only do zero grad for linear modules. 
        #If called on activation modules (like tanh), that does not have it specified,
        #it should call the function definition from the mother class in which case it will pass.
        for module in self.members:
            if module.param() is not None:
                module.zero_grad()
   
    def step(self):
        for module in self.members:
            if module.param() is not None:
                module.step()
                    
class ReLu(Module):
    def __init__(self):
        super(ReLu, self).__init__()
        self.current = torch.empty(1,1)
    
    def forward(self, x):
        self.current = x
        return torch.max(x,torch.zeros(x.size()))
    
    def backward(self, dl_dx):
        """
            Def of backprop : dl_ds = dl_dx * dsigma(s)
            dsigma for relu is f : (1, x>0, 
                                    0, x<0)
            this should work as it gives a logical tensor then into float
            not sure about dimensions.....
        """
        return dl_dx*(self.current>0).float()
    
class MSE(Module):
    """ SEE PRACTICAL 03 UPDATE DOC LATER"""
    def __init__(self):
        super(MSE, self).__init__()
    
    def forward(self, x, t):
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        return (x - t).pow(2).sum()
    
    def backward(self, x, t):
        #This is dl_dx. (dloss wrt to x for MSE loss is 2(x-t))
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        return 2 * (x - t)
        
class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.current = torch.empty(1,1)
        
    def forward(self, x):
        self.current = x
        return x.tanh()
    
    def backward(self, dl_dx):
        """ dtanh = 1/cosh^2"""
        return dl_dx*(1/(torch.pow(torch.cosh(self.current),2)))   
                   
class CrossEntropyLoss(Module):        
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        
    def softmax(self, x):
        """
            Computes softmax with shift to be numerically stable for
            large numbers or floats takes exp(x-max(x)) instead of exp(x)
        """
            #this is really stablesoftmax(x)
            #rather than softamx(x)
        z = x - x.max()
        exps = torch.exp(z)
        return (exps/torch.sum(exps))
    
    def forward(self, x, t):
        """
            With pj = exp(aj)/sum(exp(ak))
            Loss = -Sum_j (yj) log(pj), 
            with t the target being the y in the formula
            and pj = softmax(x)_j
            log(p)*t does the element-wise product then we sum
        """
        p = self.softmax(x)
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        sumResult = -torch.sum(torch.log(p)*t)
        return sumResult
    
    def backward(self, x, t):
        """
            computes dLoss 
            dl/dx_i = pi-yi from the slides
        """
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        p = self.softmax(x)
        return p - t
    
def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp
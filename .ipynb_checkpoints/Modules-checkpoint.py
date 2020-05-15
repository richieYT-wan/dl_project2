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
    
    def zero_grad(self):
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
        self.w = torch.empty(in_dim,out_dim).normal_(0,std)
        self.b = torch.empty(out_dim).normal_(0,std)
        self.dw = torch.zeros(in_dim,out_dim)
        self.db = torch.zeros(out_dim)
        self.current = torch.empty(1,1)
    
    def forward(self,x):
        """
            results of FW pass of layer, returns s = x*w + b
            using formula from Practical 03
        """
        self.current = x
        return x.mm(self.w)+self.b
   
    def backward(self, dl_ds):
        self.dw = self.current.t().mm(dl_ds)
        self.db = dl_ds.mean(0) #this works.
        return dl_ds.mm(self.w.t())
    
    def zero_grad(self):
        """
            Function to set grad to zero. Used in gradient step
        """
        self.dw.zero_()
        self.db.zero_()
    
    def param(self):
        return [[self.w, self.dw], [self.b, self.db]]
    
    def reset_param(self):
        """
            Re-initialize the weights. This "unlearns" the weights to re-use 
            the same instance across runs without having to make new instances
        """
        std = math.sqrt(2./(self.in_dim+self.out_dim))
        self.w = torch.empty(self.in_dim,self.out_dim).normal_(0,std)
        self.b = torch.empty(self.out_dim).normal_(0,std)
        self.dw = torch.zeros(self.in_dim,self.out_dim)
        self.db = torch.zeros(self.out_dim)

        
#ACTIVATION FUNCTIONS
class ReLu(Module):
    def __init__(self):
        super(ReLu, self).__init__()
        self.current = torch.empty(1,1)
    
    def forward(self, s):
        self.current = s
        return torch.max(s,torch.zeros(s.size()))
    
    def backward(self, dl_dx):
        """
            Def of backprop : dl_ds = dl_dx * dsigma(s)
            dsigma for relu is f : (1, x>0, 
                                    0, x<0)
            this should work as it gives a logical tensor then into float
        """
        
        return dl_dx*(self.current>0).float()
        
class Tanh(Module):
    def __init__(self):
        super(Tanh, self).__init__()
        self.current = torch.empty(1,1)
        
    def forward(self,s):
        self.current = s
        return s.tanh()
    
    def backward(self,dl_dx):
        """ dtanh = 1/cosh^2"""
        return dl_dx* (1/(torch.pow(torch.cosh(self.current),2)))   
               
class Softmax(Module):
    def __init__(self):
        super(Softmax,self).__init__()
        self.current = torch.empty(1,1)
        
    def forward(self,s):
        """
            Computes softmax with shift to be numerically stable for
            large numbers or floats takes exp(x-max(x)) instead of exp(x)
        """
        maxVal,_ =torch.max(s,1,keepdim=True)
        z = s- maxVal
        self.current=s #Should this be "= z" instead?
        exps = torch.exp(z)
        Sum = exps.sum(1,keepdim=True)
        return (exps/Sum)
    
    def backward(self,dl_dx):
        
        return dl_dx*(self.forward(self.current)-(self.forward(self.current)).pow(2))
    
#----------------Loss criterions---------------
class MSE(Module):
    """ SEE PRACTICAL 03 UPDATE DOC LATER"""
    def __init__(self):
        super(MSE, self).__init__()
    
    def forward(self,x,t):
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        return (x - t).pow(2).sum()
    
    def backward(self, x, t):
        #This is dl_dx. (dloss wrt to x for MSE loss is 2(x-t))
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        return 2 * (x - t)
    

class CrossEntropyLoss(Module):        
    def __init__(self):
        Module.__init__(self)
        
    def softmax(self,x):
        """
            Computes softmax with shift to be numerically stable for
            large numbers or floats takes exp(x-max(x)) instead of exp(x)
        """
        maxVal,_ =torch.max(x,1,keepdim=True)
        exps = torch.exp(x- maxVal)
        return exps/(exps.sum(1,keepdim=True))
    
    def forward(self,x,t):
        """
            With pj = exp(aj)/sum(exp(ak))
            Loss = -Sum_j (yj) log(pj), 
            with t the target being the y in the formula
            and pj = softmax(x)_j
            log(p)*t does the element-wise product then we sum
        """
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        return -torch.sum((self.softmax(x).log())*t)
        
    
    def backward(self,x,t):
        """
            computes dLoss 
            dl/dx_i = pi-yi from the slides
        """
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        return (self.softmax(x)-t)

def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp


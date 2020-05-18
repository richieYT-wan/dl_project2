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
    
#-------Fully Connected Layer-------#
class Linear(Module):
    """
        One fully connected layer.
        Input : input_dimension, output_dimension [int]
        Output : none
        Those corresponds to the numbers of nodes in a given input layer
        and the number of nodes in the output layer
        Weights and Bias are initialized using the Xavier initialization
        self.current to save inputs in memory for backprop.
        Gradients are saved in dw, db and initialized at 0.
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
            input : x [Tensor]
            output : s = x*w +b [Tensor]
            Returns the result of FW pass of a linear layer, saves input in memory 
        """
        self.current = x
        return x.mm(self.w)+self.b
   
    def backward(self, dl_ds):
        """
            Input : dl_ds [Tensor]
            output : dl_dx of previous layer (dl_dx-1) [Tensor]
            Updates the gradients according to the back-prop rules.
        """
        self.dw = self.current.t().mm(dl_ds)
        self.db = dl_ds.mean(0)
        return dl_ds.mm(self.w.t())
    
    def zero_grad(self):
        """
            Function to set grad to zero. Used in gradient step
        """
        self.dw.zero_()
        self.db.zero_()
    
    def param(self):
        """
            Input : none
            Output : The parameters of the linear layer as pairs of parameters and its gradient
        """
        return [[self.w, self.dw], [self.b, self.db]]
    
    def reset_param(self):
        """
            Re-initialize the weights. This "unlearns" the weights to re-use 
            the same instance across runs without having to remake a Linear object.
        """
        std = math.sqrt(2./(self.in_dim+self.out_dim))
        self.w = torch.empty(self.in_dim,self.out_dim).normal_(0,std)
        self.b = torch.empty(self.out_dim).normal_(0,std)
        self.dw = torch.zeros(self.in_dim,self.out_dim)
        self.db = torch.zeros(self.out_dim)

        
#-------Activation Functions-------#
"""
    dl_dx : Derivative of loss wrt x
    dl_ds : Derivative of loss wrt s
    dsigma : Derivative of the activation function
"""
class ReLu(Module):
    """
        Defines ReLU as an activation function.
    """
    def __init__(self):
        super(ReLu, self).__init__()
        self.current = torch.empty(1,1)
    
    def forward(self, s):
        """
            Input : s [Tensor]
            Output : x = sigma(s) = max(0,x) [Tensor]
        """
        self.current = s
        return torch.max(s,torch.zeros(s.size()))
    
    def backward(self, dl_dx):
        """
            Input : dl_dx [Tensor]
            Output : dl_ds [Tensor]
            
            Def of backprop : dl_ds = dl_dx * dsigma(s)
            dsigma for relu is f : (1, x>0, 
                                    0, x<0)
            done using logical tensor converted to float.
        """
        
        return dl_dx*(self.current>0).float()
        
class Tanh(Module):
    """
        Defines Tanh as an activation function.
    """
    def __init__(self):
        super(Tanh, self).__init__()
        self.current = torch.empty(1,1)
        
    def forward(self,s):
        """
            Input : s [Tensor]
            Output : x = sigma(s) = tanh(s) [Tensor]
        """
        self.current = s
        return s.tanh()
    
    def backward(self,dl_dx):
        """
            Input : dl_dx [Tensor]
            Output : dl_ds [Tensor] 
            Def of backprop : dl_ds = dl_dx * dsigma(s)
            dtanh = 1/cosh^2
        """
        return dl_dx* (1/(torch.pow(torch.cosh(self.current),2)))   
               
class Softmax(Module):
    """
        Defines Softmax as an activation function.
        Usually used in the last layer.
    """
    def __init__(self):
        super(Softmax,self).__init__()
        self.current = torch.empty(1,1)
        
    def forward(self,s):
        """
            Input : s [Tensor]
            Output : x = sigma(s) = softmax(s) [Tensor]
            
            Computes softmax with shift to be numerically stable for
            large numbers or floats. Takes exp(x-max(x)) instead of exp(x)
        """
        maxVal,_ =torch.max(s,1,keepdim=True)
        z = s- maxVal
        self.current=s 
        exps = torch.exp(z)
        Sum = exps.sum(1,keepdim=True)
        return (exps/Sum)
    
    def backward(self,dl_dx):
        """
            Input : dl_dx [Tensor] 
            output : dl_ds 
            Def of backprop : dl_ds = dl_dx* dsigma(s)            
        """
        return dl_dx*(self.forward(self.current)-(self.forward(self.current)).pow(2))
    

#-------Loss Criterions-------#
class MSE(Module):
    """
        Defines the MSE loss (L2 criterion)
    """
    def __init__(self):
        super(MSE, self).__init__()
    
    def forward(self,x,t):
        """
            Input : Prediction x, target t (as hot labels) [Tensor]
            Output : Loss evaluated for x given t [Float]
        """
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        return (x - t).pow(2).sum()
    
    def backward(self, x, t):
        """
            Input : Prediction x, target t (as hot labels) [Tensors]
            Output : dl_dx
            Returns the derivative of the MSE loss wrt x
            dl_dx = 2(x-t)
        """
        #This is dl_dx. (dloss wrt to x for MSE loss is 2(x-t))
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        return 2 * (x - t)
    

class CrossEntropyLoss(Module): 
    """
        Defines CrossEntropyLoss as a criterion
    """
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
            Input : prediction x, target t (as hot labels) [Tensors]
            Output : CrossEntropyLoss as defined below [Float]
            
            With p_j = exp(x_j)/sum(exp(x_j)) = softmax(x_j)
            Loss = -Sum_j (yj) log(pj), 
            
            log(p)*t does the element-wise product then we sum
        """
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        return -torch.sum((self.softmax(x).log())*t)
        
    
    def backward(self,x,t):
        """
            Input : prediction x, target t (as hot labels) [Tensors]
            Output : dl_dx
            Returns the derivative of loss wrt X
            
            dl/dx_i = pi-yi 
        """
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        return (self.softmax(x)-t)

def convert_to_one_hot_labels(input, target):
    """
        Input : Target [tensor], size : n_samples x 1
        Output : Target as hot labels [Tensor], size : n_samples x 2
        
        Function to convert targets to hot labels.
    """
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp


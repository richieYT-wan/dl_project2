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
        self.w = torch.empty(in_dim,out_dim).normal_(0,std)
        self.b = torch.empty(out_dim).normal_(0,std)
        self.dw = torch.zeros(in_dim,out_dim)
        self.db = torch.zeros(out_dim)
    
    def param(self):
        return [[self.w, self.dw], [self.b, self.db]]
        
    def forward(self,x):
        """
            results of FW pass of layer, returns s = x*w + b
            using formula from Practical 03
        """
        s = x.mm(self.w)+self.b
        return s
    
    def backward(self,x,dl_ds):
        """
            Backward pass, see practical 03 and slide 9,11 of course 3.6
            Accumulates gradient according to chain rule, with mat product.
            dl_ds is the derivative of loss wrt to s = x*w + b
            dl_ds = dl_dx.mul(sigma'(s)) which is given to him the
            backward of activation functions.
        """
        self.dw.add_(dl_ds.view(-1, 1).mm(x.view(1, -1)))
        self.db.add_(dl_ds.view(-1))
        dl_dx_prev = self.w.t().mm(dl_ds)
        return dl_dx_prev
    
    def zero_grad(self):
        """
            Function to set grad to zero. Used in gradient step
        """
        self.dw.zero_()
        self.db.zero_()
        
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
        
        self.parameters = []
        for module in self.members:
            if module.param() is not None:
                for p in module.param():
                    self.parameters.append(p)
        
        self.memory = []
        self.lossmemory = []
        
    def param(self):
        
        return self.parameters
    
    def forward(self,x):
        #MUST ADD THINGS TO MEMORY AFTER OPERATION.
        # x is the input to which the operations of each module must be applied in the right order.
        #Does this need to save the first entry ?
        #It should save the value at each step (ex what is x1, s1, sigma(s1), etc.)
        #this is done in the 
        #Should backprop also have another memory to save the backprop'd gradient and what else? Loss??
        
        for module in self.members:
            x = module(x)
            self.memory.append(x)
        return x
        
    def backward(self, out, dl):
        #must enumerate backwards in the list in memory. 
        #what should this do ?
        #should save all the losses at each operation?
        #backward (from Module) takes as argument : Z, dloss, where z is a
        #can use for reversed(...) or use for i in () then use memory[-1-i] to iterate backward.
        self.lossmemory.append(dl)
        for idx, module in reversed(enumerate(self.members)):
            #the input for backward is the value saved in memory 
            dl = module.backward(self.memory[i],dl)
            self.lossmemory.append(dl)
        #reversed loss memory
        return self.lossmemory[::-1]
    
    def zero_grad(self):
        #should only do zero grad for linear modules. 
        #If called on activation modules (like tanh), that does not have it specified,
        #it should call the function definition from the mother class in which case it will pass.
        for module in self.members :
            module.zero_grad()
        
#ACTIVATION FUNCTIONS
class ReLu(Module):
    def __init__(self):
        super(ReLu, self).__init__()
    
    def forward(self,x):
        return torch.max(x,torch.zeros(x.size()))
    
    def backward(self,s,dl_dx):
        """
            Def of backprop : dl_ds = dl_dx * dsigma(s)
            dsigma for relu is f : (1, x>0, 
                                    0, x<0)
            this should work as it gives a logical tensor then into float
            not sure about dimensions.....
        """
        return dl_dx * (s>0).float()
    
class MSE(Module):
    """ SEE PRACTICAL 03 UPDATE DOC LATER"""
    def __init__(self):
        super(MSE, self).__init__()
    
    def forward(self,x,t):
        t = convert_to_one_hot_labels(torch.tensor([0, 1]), t)
        return (x - t).pow(2).sum()
    
    def backward(self, x, t):
        #This is dl_dx. (dloss wrt to x for MSE loss is 2(x-t))
        return 2 * (x - t)
    # MAYBEWRONG UPDATE THIS LATER
        
#class Tanh(Module):
#    def __init__(self):
#        super(Tanh, self).__init__()
#        
#    def forward(self,x):
#        return x.tanh()
#    
#    def backward(self,s,dl_dx):
#        """ dtanh = 1/cosh^2"""
#        return dl_dx* (1/(torch.pow(torch.cosh(s),2)))   
#               
#    
#class CrossEntropyLoss(Module):        
#    def __init__(self):
#        Module.__init__(self)
#        
#    def softmax(x):
#        """
#            Computes softmax with shift to be numerically stable for
#            large numbers or floats takes exp(x-max(x)) instead of exp(x)
#        """
#            #this is really stablesoftmax(x)
#            #rather than softamx(x)
#        z = x- x.max()
#        exps = torch.exp(z)
#        return (exps/torch.sum(exps))
#    
#    def forward(self,x,t):
#        """
#            With pj = exp(aj)/sum(exp(ak))
#            Loss = -Sum_j (yj) log(pj), 
#            with t the target being the y in the formula
#            and pj = softmax(x)_j
#            log(p)*t does the element-wise product then we sum
#        """
#        p = self.softmax(x)
#        sumResult = -torch.sum(torch.log(p)*t)
#        return sumResult
#    
#    def backward(self,x,t):
#        """
#            computes dLoss 
#            dl/dx_i = pi-yi from the slides
#        """
#        p = self.softmax(x)
#        return p-t
#    
def convert_to_one_hot_labels(input, target):
    tmp = input.new_zeros(target.size(0), target.max() + 1)
    tmp.scatter_(1, target.view(-1, 1), 1.0)
    return tmp
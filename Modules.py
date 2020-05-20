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

class Sequential(Module):
    """
        An instance of the sequential class contains the Modules passed to it as arguments.
        For example, pass linear, relu to it, it will be stored within its attribute "members"
        the attribute memory is used to save the operations in the order in which they were done
        to allow backprop. It also stores the parameters of its content in its own self.parameters.
    """
    def __call__(self, *input):
        return self.forward(*input)
    
    def __init__(self,*Modules):
        super(Sequential, self).__init__()
        self.members = Modules
        self.parameters = []
        for module in self.members:
            if module.param() is not None:
                for p in module.param():
                    self.parameters.append(p)
        self.memory = []
        
    def param(self):
        return self.parameters
    
    def forward(self, x):
        """
            Input : x, the dataset [Tensor]
            Output : x, the result of all operations applied sequentially [Tensor]
            
            Calls the forward of each module and applies it on the input x, in the forward order.
            Saves results of each operation in memory (ex what is x1, s1, sigma(s1), etc.)
            using the module's .current attribute
        """
        
        for index, module in enumerate(self.members):
            #re-updates the modules' parameters, circumvents issues created by the 
            #backward which re-updates the parameters (to store gradient in parameters of Sequential)
            if module.param() is not None : 
                module.w = self.param()[index][0]
                module.b = self.param()[index+1][0]

            x = module(x)
            self.memory.append(module.current)
        return x
        
    
    def backward(self, dl):
        """
            Input : derivative of loss wrt x
            Output : none
            
            Takes the dloss, and runs it across list of modules backward 
            Propagates the dloss backward and updates it using the .backward function
            of each module in the list of members.
            
        """
        rev = self.members[::-1]
        revMemory = self.memory[::-1]
        revParam = []
        
        for index,module in enumerate(rev):
            #The module.current update circumvents issues when using same activation (ex RELU) multiple times 
            #with the same ReLU instance/object across a single sequential instance
            #Ex : Model = sequential(linear1,relu,linear2,relu,linear3,relu)
            module.current = revMemory[index]
            dl = module.backward(dl)
            #As it propagates backward, the gradients dw, db of each linear modules are added back
            #to the parameters of a Sequential object, allowing them to be properly updated by Optimizer.py
            if module.param() is not None:
                #Add the db,b,dw,w in this order because at the end we reverse.
                for p in reversed(module.param()):
                    revParam.append(p)
        #updates the parameters returned by sequential.
        self.parameters = revParam[::-1]
             
    def zero_grad(self):
        """
            Calls the zero_grad function of each modules in its members list.
            Only Linear modules have zero_grad and will reset.
        """
        for module in self.members :
            module.zero_grad()
    
    def reset_param(self):
        """
            Resets the parameters of each module in its member list.
            Effectively "unlearns" and resets the weights and bias.
            Useful when running the same Sequential (and linear) objects multiple times without re-instancing new objects.
        """
        self.parameters = []
        for module in self.members:
            if module.param() is not None:
                module.reset_param()
                for p in module.param():
                    self.parameters.append(p)    
                    

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

        
#ACTIVATION FUNCTIONS
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
    

#----------------Loss criterions---------------
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


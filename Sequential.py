import torch
import numpy as np
import math

class Sequential():
    """
        An instance of the sequential class contains the Modules passed to it as arguments.
        For example, pass linear, relu to it, it will be stored within its attribute "members"
        the attribute memory is used to save the operations in the order in which they were done
        to allow backprop (?)
    """
    def __init__(self,*Modules):
        self.members = []
        self.params = []
        for mod in Modules:
            self.members.append(mod)
            self.parameters.extend(mod.param())
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
        out = x
        for module in self.members :
            out = module.forward(out)
            self.memory.append(out)
        return out
        
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
            
    
class Sequential():
    """
        An instance of the sequential class contains the Modules passed to it as arguments.
        For example, pass linear, relu to it, it will be stored within its attribute "members"
        the attribute memory is used to save the operations in the order in which they were done
        to allow backprop (?)
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
        self.lossmemory = []
        
    def param(self):
        return self.parameters
    
    def forward(self, x):
        #MUST ADD THINGS TO MEMORY AFTER OPERATION.
        # x is the input to which the operations of 
        #each module must be applied in the right order.
        #It should save the value at each step (ex what is x1, s1, sigma(s1), etc.)
        #this is done using self.memory and module.current
        
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
        #for module in self.members:
            
        rev = self.members[::-1]
        revMemory = self.memory[::-1]
        revParam = []
        for index,module in enumerate(rev):
            #Circumvents issues when using same activation (ex RELU) multiple times 
            #with the same ReLU instance/object.
            module.current = revMemory[index]
            dl = module.backward(dl)
            if module.param() is not None:
                #here I add the db,b,dw,w in this order because at the end we reverse.
                for p in reversed(module.param()):
                    revParam.append(p)
        #updates the parameters returned by sequential.
        self.parameters = revParam[::-1]
         
        
    
    def zero_grad(self):
        #should only do zero grad for linear modules. 
        #If called on activation modules (like tanh), that does not have it specified,
        #it should call the function definition from the mother class in which case it will pass.
        for module in self.members :
            module.zero_grad()
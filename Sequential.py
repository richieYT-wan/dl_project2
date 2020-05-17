class Sequential():
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
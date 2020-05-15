import matplotlib.pyplot as plt
import torch
import math
import Modules
import Optimizer
import Sequential

def plot_data(data, labels):
        
    data_out = data[torch.where(labels==1)]
    data_in = data[torch.where(labels==0)]
    
    plt.figure(figsize=(6,6))
    
    plt.scatter(data_out[:,0],data_out[:,1], color='k')
    plt.scatter(data_in[:,0], data_in[:,1], color='r')
    plt.show()
    
def generate_disc_set(nb_sample, show_data=False):
    
    data = torch.empty(nb_sample, 2).uniform_(0, 1)
    labels = data.sub(0.5).pow(2).sum(1).sub(1/(2*math.pi)).sign().add(1).div(2).long()
    
    data_test = torch.empty(nb_sample, 2).uniform_(0, 1)
    labels_test = data_test.sub(0.5).pow(2).sum(1).sub(1/(2*math.pi)).sign().add(1).div(2).long()
    
    if show_data:
        plot_data(data, labels)

    return data, labels, data_test, labels_test

def train_model_SGD(model, criterion, 
                    train_input, train_target, test_input, test_target,
                    mini_batch_size, nb_epochs, eta = 1e-4, wd = None, 
                    momentum = False, adaptive = False,
                    plot_loss = False, plot_points = False):
    """
        Function to train a given model, criterion.
        Inputs: 
            model : a Sequential object containing its modules [Sequential]
            criterion : Criterion used for loss. (Either MSE or CrossEntropyLoss) [Loss module]
            train_input, train_target : Input tensors and target tensors for training [Tensors]
            test_input, test target : Input tensors and target tensors for validation [Tensors]
            
            mini_batch_size, nb_epochs : Batch size and epochs used for training [int]
            eta, wd : Learning rate, Weight Decay used for training [float]
            
            Adaptive : Decay of learning rate over nb_epochs/5 [float]
                       Used to have an adaptive learning rate which decays to stabilize training.
                       Typical value for adaptive : 0.7 to 0.9
                       Allows faster training and test error to stabilize.
                       
            plot_loss : Whether the loss and accuracy should be plotted at the end. [Bool] 
            plot_points : Whether the data_points and predictions should be plotted [Bool]
                
    """
    losses = []
    test_accs = []
    for e in range(nb_epochs):
        
        #train
        sum_loss = 0
        if adaptive:
            #Every nb_epochs/5, the learning rate is divided
            epochs_drop = nb_epochs/5
            eta = eta * math.pow(adaptive,math.floor((1+e)/epochs_drop))
        
        #Random sampling
        for b in list(torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(train_input.size(0))),batch_size=mini_batch_size, drop_last=False)):
            output = model(train_input[b])
            loss = criterion(output, train_target[b])
            model.zero_grad()
            model.backward(criterion.backward(output, train_target[b]))
            if not momentum:
                optim = Optimizer.SGD(model.param(),eta=eta,wd=wd)
            if momentum:
                optim = Optimizer.SGD_momentum(model.param(),eta=eta,wd=wd,gamma=momentum)
            optim.step()
            sum_loss += loss.item()
    
        losses.append(sum_loss)
        
        #test
        sum_error = 0
        predicted_test_classes = []
        data_plot = []
        for t in list(torch.utils.data.BatchSampler(torch.utils.data.RandomSampler(range(test_input.size(0))), batch_size=mini_batch_size, drop_last=False)):
            test_out = model(test_input[t])
            predicted_test_batch_classes = test_out.max(1)[1]
            nb_errors = torch.where(test_target[t] != predicted_test_batch_classes)[0].size(0)
            
            # plot
            sum_error += nb_errors
            predicted_test_classes.append(predicted_test_batch_classes)
            data_plot.append(test_input[t])
        
        test_accs.append((100 - ((100 * sum_error) / test_input.size(0))))
        
        #plot
        if plot_points:
            predicted_test_classes = torch.cat(predicted_test_classes, dim=0)
            data_plot = torch.cat(data_plot, dim=0)
        
        if e%int((nb_epochs/10)) == 0 or e == nb_epochs-1:
            print('{:d} train_loss {:.02f} test_error {:.02f}%'.format(e, sum_loss, (100 * sum_error) / test_input.size(0)))
            if plot_points:
                plot_data(data_plot, predicted_test_classes)
        
    # Train loss and test acc plot 
    if plot_loss:
        title = str(criterion)[str(criterion).find('.')+1:str(criterion).find(" ")]
        title = "Criterion : "+title+" Adaptive: {}".format(adaptive)
        title2 = str(model.members[-1])[str(model.members[-1]).find('.')+1:str(model.members[-1]).find(" ")]
        title2 = "\n Final activation :"+title2
        title+title2
        fig, ax1 = plt.subplots(figsize=(6,6))
    
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Train loss')
        plt1 = ax1.plot(range(len(losses)), losses, color='tab:red', label='Train loss')
        ax1.tick_params(axis='y')
    
        ax2 = ax1.twinx()  
        ax2.set_ylabel('Test accuracy [%]')  
        plt2 = ax2.plot(range(len(test_accs)), test_accs, color='tab:blue', label='Test accuracy')
        ax2.tick_params(axis='y')
    
        plts = plt1+plt2
        labs = [p.get_label() for p in plts]
        plt.legend(plts, labs, loc='center right')
        plt.title(title+title2)
        plt.show()
    return losses, test_accs
            



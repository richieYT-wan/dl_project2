import matplotlib.pyplot as plt
import torch
import math
import Modules
import Optimizer
import os
def plot_data(data, labels):
    """
        Plots all the points of the dataset.
        Points with label 1 are inside a circle of radius 1/sqrt(2pi) centered at (0.5,0.5) and in red.
        Points with label 0 are outside the circle and in black.
    """
    data_out = data[torch.where(labels==0)]
    data_in = data[torch.where(labels==1)]

    plt.figure(figsize=(6,6))

    plt.scatter(data_out[:,0],data_out[:,1], color='k', label='label 0 - outside')
    plt.scatter(data_in[:,0], data_in[:,1], color='r', label='label 1 - inside')
    plt.legend(bbox_to_anchor=(0.5, 1.13), loc='upper center')
    plt.show(block=False)

def generate_disc_set(nb_sample, show_data=False):
    """
        Generates a disc data_set.
        Input : nb_sample [int]
        output : Training Dataset composed of nb_samples point uniformly distributed between 0,1 
                 Training Labels of size nb_samples, = 0 or 1
                 Test dataset and its label.
    """
    data = torch.empty(nbsample, 2).uniform(0, 1)
    labels = (data.sub(0.5).pow(2).sum(1) <= (1/(2math.pi))).long()

    data_test = torch.empty(nbsample, 2).uniform(0, 1)
    labels_test = (data_test.sub(0.5).pow(2).sum(1) <= (1/(2math.pi))).long()

    if show_data:
        plot_data(data, labels)

    return data, labels, data_test, labels_test


def train_model_SGD(model, criterion, 
                    train_input, train_target, test_input, test_target,
                    mini_batch_size=50, nb_epochs=500, eta = 1e-4, wd = 1e-7, adaptive = False,
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
    # Note : The optimizer is initialized at each epoch with parameters of the model 
    # after backward due to issues with the implementation of how parameters are accessed in memory
    # There was an issue where the gradient was updated in the backward, but since Optimizer
    # was initialized before the backward, it had a copy of the parameters with null gradients
    # as attribute and therefore performed the gradient step with 0 change.
    for e in range(nb_epochs):
        
        #train
        sum_loss = 0
        if adaptive:
            #Every nb_epochs/5, the learning rate is divided
            epochs_drop = nb_epochs/5
            eta = eta * math.pow(adaptive,math.floor((1+e)/epochs_drop))
        
        #Random sampling
        indexes = torch.randperm(train_input.size(0)).tolist()
        for b in list(indexes[x:x+mini_batch_size] for x in range(0, len(indexes), mini_batch_size)):
            output = model(train_input[b])
            loss = criterion(output, train_target[b])
            model.zero_grad()
            model.backward(criterion.backward(output, train_target[b]))
            optim = Optimizer.SGD(model.param(),eta=eta,wd=wd)
            optim.step()
            sum_loss += loss.item()
        
        sum_loss = sum_loss/train_input.size(0)
        losses.append(sum_loss)
        
        #test
        sum_error = 0
        predicted_test_classes = []
        data_plot = []
        indexes = torch.randperm(test_input.size(0)).tolist()
        for t in list(indexes[x:x+mini_batch_size] for x in range(0, len(indexes), mini_batch_size)):
            test_out = model(test_input[t])
            predicted_test_batch_classes = test_out.max(1)[1]
            nb_errors = torch.where(test_target[t] != predicted_test_batch_classes)[0].size(0)
            
            # plot
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

        if not os.path.exists('../figures'):
            os.makedirs('../figures')
        
        fig.savefig('../figures/trainloss_testacc.png', dpi=600)
        
    return losses, test_accs
            
def users_choices():
    print('First figure shows the training set.')
    print('\n')

    # offer to the user the choice to choose his parameters
    choose = input("Do you want to choose your model [yes/no] (if not our best model will run)?\n> ")
    while (choose != 'yes' and choose != 'no'):
        print('ERROR: your answer is not valid!')
        choose = input('Please choose again between [yes/no].\n>')        
    if (choose == 'yes'):
        print("OK: Let's choose!")
    if (choose == 'no'):
        print('OK: Our best model will run')   
        act_fun='tanh'
        criter='cross entropy' 
        lr=4e-3
        adapt_lrd=0.9 
        wd=1e-6
        plot=True
        plot_classif=False
        print('\n')
        print('>>> MODEL RUNNING:')  
        print('\n', 'Last layer activation function:', act_fun, '\n', 'Loss criterion:', criter, '\n', 'Learning rate:', lr, '\n', 
                'Adaptive learning rate:', adapt_lrd, '\n', 'Weight decay:', wd, '\n')
        return act_fun, criter, lr, adapt_lrd, wd, plot, plot_classif
    
    print('\n')


    # choose activation function
    act_fun = input("Please choose an activation function for the last layer of the model [relu/softmax/tanh].\n> ")
    while (act_fun != 'relu' and act_fun != 'softmax' and act_fun != 'tanh'):
        print('ERROR:', act_fun, 'is not valid!')
        act_fun = input('Please choose again between [relu/softmax/tanh].\n> ')        
    if (act_fun == 'relu' or act_fun == 'softmax' or act_fun == 'tanh'):
        print('OK:', act_fun, 'is selected as activation function for the last layer.')
    
    print('\n')

    # choose criterion (loss)
    criter = input("Please choose a loss criterion [cross entropy/MSE].\n> ")
    while (criter != 'cross entropy' and criter != 'MSE'):
        print('ERROR:', criter, 'is not valid!')
        criter = input('Please choose again between [cross entropy/MSE].\n> ')       
    if (criter == 'cross entropy' or criter == 'MSE'):
        print('OK:', criter, 'is selected as loss criterion.')

    print('\n')

    # choose w/ or w/o adaptive lr
    if(act_fun == 'tanh' or act_fun == 'softmax'):
        adapt_lrd = input("Do you want to use adaptive learning rate decay [yes/no]?\n> ")
        while (adapt_lrd != 'yes' and adapt_lrd != 'no'):
            print('ERROR: your answer is not valid!')
            adapt_lrd = input('Please choose again between [yes/no].\n> ')        
        if (adapt_lrd == 'yes'):
            adapt_lrd = 0.9
            print('OK: adaptive learning rate will be used with a value of 0.9.')
        if (adapt_lrd == 'no'):
            adapt_lrd = False
            print('OK: adapative learning rate decay won\'t be used.')  

    print('\n')
    
    # set parameters depending on choices
    if(act_fun == 'tanh'):
        if(criter == 'MSE'):
            if(adapt_lrd == 0.9):
                lr = 3e-3
                wd = 2e-7
            if(adapt_lrd == False):
                lr = 5e-3
                wd = 3e-7
        if(criter == 'cross entropy'):
            lr = 3e-3
            wd = 2e-7

    if(act_fun == 'softmax'):
        lr = 3e-3
        wd = 2e-7

    if(act_fun == 'relu'):
        adapt_lrd = None
        lr = 7e-4
        wd = 1e-7
    
    # choose to display or not plots of loss & accuracy vs epochs
    plot = input("Do you want plot (train loss + test accuracy VS nb epochs) to be displayed [yes/no]?\n> ")
    while (plot != 'yes' and plot != 'no'):
        print('ERROR: your answer is not valid!')
        plot = input('Please choose again between [yes/no].\n> ')        
    if (plot == 'yes'):
        print('OK: plot will be displayed')
    if (plot == 'no'):
        plot = False
        print('OK: plot won\'t be displayed')  

    print('\n')
    
    # choose to display or not result classification of points
    plot_classif = input("Do you want classified points to be displayed [yes/no]?\n> ")
    while (plot_classif != 'yes' and plot_classif != 'no'):
        print('ERROR: your answer is not valid!')
        plot_classif = input('Please choose again between [yes/no].\n> ')        
    if (plot_classif == 'yes'):
        print('OK: plots will be displayed')
    if (plot_classif == 'no'):
        plot_classif = False
        print('OK: plots won\'t be displayed') 

    print('\n')
    print('>>> MODEL RUNNING:')  
    print('\n', 'Last layer activation function:', act_fun, '\n', 'Loss criterion:', criter, '\n', 'Learning rate:', lr, '\n', 
                'Adaptive learning rate:', adapt_lrd, '\n', 'Weight decay:', wd, '\n')

    return act_fun, criter, lr, adapt_lrd, wd, plot, plot_classif

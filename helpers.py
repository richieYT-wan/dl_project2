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
    return losses, test_accs
            
def users_choices():
    # offer to the user the choice to choose his parameters
    choose = input("Do you want to choose parameters [yes/no] (if not our best model will run)?\n>")
    while (choose != 'yes' and choose != 'no'):
        print('ERROR: your answer is not valid!')
        choose = input('Please choose again between [yes/no].\n>')        
    if (choose == 'yes'):
        print("OK: Let's choose!")
    if (choose == 'no'):
        print('OK: Our best model will be run')   
        act_fun='softmax'
        criter='cross entropy' 
        lr=5e-2
        adapt_lrd=0.9 
        wd=1e-7
        plot=True
        plot_classif=False
        print('\n', 'Last layer activation function:', act_fun, '\n', 'Loss criterion:', criter, '\n', 'Learning rate:', lr, '\n', 
                'Adaptive learning rate:', adapt_lrd, '\n', 'Weight decay:', wd, '\n')
        return act_fun, criter, lr, adapt_lrd, wd, plot, plot_classif
    
    print('\n')


    # choose activation function
    act_fun = input("Please choose an activation function for the last layer of the model [relu/softmax/tanh].\n>")
    while (act_fun != 'relu' and act_fun != 'softmax' and act_fun != 'tanh'):
        print('ERROR:', act_fun, 'is not valid!')
        act_fun = input('Please choose again between [relu/softmax/tanh].\n>')        
    if (act_fun == 'relu' or act_fun == 'softmax' or act_fun == 'tanh'):
        print('OK:', act_fun, 'is selected as activation function for the last layer.')
    
    print('\n')

    # choose criterion (loss)
    criter = input("Please choose a loss criterion [cross entropy/MSE].\n>")
    while (criter != 'cross entropy' and criter != 'MSE'):
        print('ERROR:', criter, 'is not valid!')
        criter = input('Please choose again between [cross entropy/MSE].\n>')       
    if (criter == 'cross entropy' or criter == 'MSE'):
        print('OK:', criter, 'is selected as loss criterion.')

    print('\n')

    # choose learning rate
    lr = input("Please choose a learning rate in the range [1e-7, 10].\n>")
    while (float(lr) > 10 or float(lr) < 1e-7):
        print('ERROR:', lr, 'is not valid!')
        lr = input('Please choose again between in the range [1e-7, 10].\n>')       
    if (float(lr) <= 10 and float(lr) >= 1e-7):
        lr = float(lr)
        print('OK:', lr, 'is selected as learning rate.')

    print('\n')
    
    # choose adaptive learning rate or not
    adapt_lrd = input("Please choose a decay value for adaptive learning rate decay in the range [1e-3, 10] or write 'no'.\n>")
    if (adapt_lrd == 'no'):
        adapt_lrd = False
        print('OK: adapative learning rate decay won\'t be used.') 
    if (adapt_lrd != False):
        while (float(adapt_lrd) > 10 or float(adapt_lrd) < 1e-3):
            print('ERROR:', adapt_lrd, 'is not valid!')
            adapt_lrd = input("Please choose again in the range [1e-3, 10] or write 'no'.\n>")   
            if (adapt_lrd == 'no'):
                adapt_lrd = False
                print('OK: adapative learning rate decay won\'t be used.') 
                break
        if (float(adapt_lrd) <= 10 and float(adapt_lrd) >= 1e-3):
            adapt_lrd = float(adapt_lrd)
            print('OK:', adapt_lrd, 'is selected as learning rate decay.')

    print('\n')
    
    # choose weight decay or not
    wd = input("Please choose a weight decay in the range [1e-10, 1] or write 'no'.\n>")
    if (wd == 'no'):
        wd = False
        print('OK: weight decay won\'t be used.') 
    if (wd != False):
        while (float(wd) > 1 or float(wd) < 1e-10):
            print('ERROR:', wd, 'is not valid!')
            wd = input("Please choose again in the range [1e-10, 1] or write 'no'.\n>")   
            if (wd == 'no'):
                wd = False
                print('OK: weight decay won\'t be used.') 
                break
        if (float(wd) <= 1 and float(wd) >= 1e-10):
            wd = float(wd)
            print('OK:', wd, 'is selected as weight decay.')    

    print('\n')
    
    # choose to display or not plots of loss & accuracy vs epochs
    plot = input("Do you want plots (train loss + test accuracy VS nb epochs) to be displayed [yes/no]?\n>")
    while (plot != 'yes' and plot != 'no'):
        print('ERROR: your answer is not valid!')
        plot = input('Please choose again between [yes/no].\n>')        
    if (plot == 'yes'):
        print('OK: plots will be displayed')
    if (plot == 'no'):
        plot = False
        print('OK: plots won\'t be displayed')  

    print('\n')
    
    # choose to display or not result classification of points
    plot_classif = input("Do you want classified points to be displayed [yes/no]?\n>")
    while (plot_classif != 'yes' and plot_classif != 'no'):
        print('ERROR: your answer is not valid!')
        plot_classif = input('Please choose again between [yes/no].\n>')        
    if (plot_classif == 'yes'):
        print('OK: plots will be displayed')
    if (plot_classif == 'no'):
        plot_classif = False
        print('OK: plots won\'t be displayed')   
 

    return act_fun, criter, lr, adapt_lrd, wd, plot, plot_classif

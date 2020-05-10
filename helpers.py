import matplotlib.pyplot as plt
import torch
import math

def plot_data(data, labels):
        
    data_out = data[torch.where(labels==1)]
    data_in = data[torch.where(labels==0)]
    
    plt.figure(figsize=(6,6))
    
    plt.scatter(data_out[:,0],data_out[:,1])
    plt.scatter(data_in[:,0], data_in[:,1])
    plt.show()
    
def generate_disc_set(nb_sample, show_data=False):
    
    data = torch.empty(nb_sample, 2).uniform_(0, 1)
    labels = data.sub(0.5).pow(2).sum(1).sub(1/(2*math.pi)).sign().add(1).div(2).long()
    
    data_test = torch.empty(nb_sample, 2).uniform_(0, 1)
    labels_test = data_test.sub(0.5).pow(2).sum(1).sub(1/(2*math.pi)).sign().add(1).div(2).long()
    
    if show_data:
        plot_data(data, labels)

    return data, labels, data_test, labels_test

def normalize_data(data, data_test):
    mean, std = data.mean(), data.std()

    data.sub_(mean).div_(std)
    data_test.sub_(mean).div_(std)
    return data, data_test

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output = model(data_input.narrow(0, b, mini_batch_size))
        _, predicted_classes = torch.max(output, 1)
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

def train_model(model, criterion, optimizer, data, target, data_test, target_test, nb_epochs, mini_batch_size):
    losses = []

    for e in range(nb_epochs):
        sum_loss = 0
        for b in range(0, data.size(0), mini_batch_size):
            output = model(data.narrow(0, b, mini_batch_size))
            
            loss = criterion.forward(output,target.narrow(0, b, mini_batch_size))
            sum_loss += loss.item()
            
            model.zero_grad()
            
            dl_dx = criterion.backward(output,target.narrow(0, b, mini_batch_size))
            model.backward(dl_dx)
            
            optimizer.step()
            
        losses.append(sum_loss)

    print('TRAIN ERROR = {:.2f}%'.format(compute_nb_errors(model, data, target, mini_batch_size) / data.size(0) * 100))
    print('TEST ERROR = {:.2f}%'.format(compute_nb_errors(model, data_test, target_test, mini_batch_size) / data_test.size(0) * 100))
    return losses



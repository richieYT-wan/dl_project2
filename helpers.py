import matplotlib.pyplot as plt
import torch
import math

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
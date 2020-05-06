import matplotlib.pyplot as plt
import torch

def plot_data(data, labels):
        
    data_out = data[torch.where(labels==1)]
    data_in = data[torch.where(labels==0)]
    
    plt.figure(figsize=(6,6))
    
    plt.scatter(data_out[:,0],data_out[:,1])
    plt.scatter(data_in[:,0], data_in[:,1])
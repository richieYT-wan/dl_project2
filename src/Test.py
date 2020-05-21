from helpers import *

torch.set_grad_enabled(False)

data, target, data_test, target_test = generate_disc_set(nb_sample=1000, show_data=True)

act_fun, criter, lr, adapt_lrd, wd, plot, plot_classif = users_choices()

# Parameters
nb_epochs = 250
mini_batch_size = 50

# Modules
m1 = Modules.Linear(2,25)
m2 = Modules.Linear(25,25)
m3 = Modules.Linear(25,25)
m4 = Modules.Linear(25,2)
relu = Modules.ReLu()
tanh = Modules.Tanh() 
softmax = Modules.Softmax()

# Model
if (act_fun == 'relu'):
    model = Modules.Sequential(m1, relu, m2, tanh, m3, tanh, m4, relu)
if (act_fun == 'tanh'):
    model = Modules.Sequential(m1, relu, m2, tanh, m3, tanh, m4, tanh)
if (act_fun == 'softmax'):
    model = Modules.Sequential(m1, relu, m2, tanh, m3, tanh, m4, softmax)

model.reset_param()

# Loss criterions
if (criter == 'cross entropy'):
    criterion = Modules.CrossEntropyLoss()
if (criter == 'MSE'):
    criterion = Modules.MSE()

# Train model
losses, test_accs = train_model_SGD(model, criterion, data, target, data_test, target_test,
                                    mini_batch_size, nb_epochs, eta=lr, wd=wd,  
                                    adaptive=adapt_lrd, plot_loss=plot, plot_points=plot_classif)
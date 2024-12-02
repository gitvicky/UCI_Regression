# %%
#Importing the necessary packages
import os 
import sys
import numpy as np
from tqdm import tqdm 
import torch
import matplotlib
import matplotlib.pyplot as plt
import time 
from timeit import default_timer
from tqdm import tqdm 

from MLP import * 
from data_loaders import * 
from utils import * 

#Setting up locations. 
file_loc = os.getcwd()
data_loc = file_loc + '/Data'
model_loc = file_loc + '/Weights/'
plot_loc = file_loc + '/Plots'

#Setting up the seeds and devices
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.float32)
# %% 
#Configuration
model_config = { "Model": 'MLP',
                 "Epochs": 500,
                 "Batch Size": 50,
                 "Optimizer": 'Adam',
                 "Learning Rate": 5e-3,
                 "Scheduler Step": 50,
                 "Scheduler Gamma": 0.5,
                 "Activation": 'Tanh',
                 "Normalisation Strategy": 'Min-Max. Variable',
                 "Layers": 4,
                 "Neurons": 256,
                 "Loss Function": 'MSE',
                 }


#Cases - residential_building, sgemm, community_crime
# case = 'sgemm'
# case = 'crimes_and_community'
case = 'bias_correction'

case_config, inputs, outputs = load_data(case)

#Setting the configuration
configuration = case_config | model_config

# %%
#Normalise Data

norm_strategy = configuration['Normalisation Strategy']

normalizer = Normalisation(norm_strategy)

in_normalizer = normalizer(inputs)
out_normalizer = normalizer(outputs)

in_norm = in_normalizer.encode(inputs)
out_norm = out_normalizer.encode(outputs)

#Train-Test Split
from sklearn.model_selection import train_test_split
in_train, in_test, out_train, out_test = train_test_split(
    in_norm, out_norm, test_size=0.33, random_state=0)

#Setting up the training and testing data splits
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(in_train, out_train), batch_size=configuration['Batch Size'], shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(in_test, out_test), batch_size=configuration['Batch Size'], shuffle=False)
# %% 
#Setting up the Model 

model = MLP(in_features=configuration['in_dims'], 
            out_features=configuration['out_dims'], 
            num_layers=configuration['Layers'], 
            num_neurons=configuration['Neurons'])

model.to(device)

# print("Number of model params : " + str(model.count_params()))

#Setting up the optimizer and scheduler, loss and epochs 
optimizer = torch.optim.Adam(model.parameters(), lr=configuration['Learning Rate'], weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=configuration['Scheduler Step'], gamma=configuration['Scheduler Gamma'])
loss_func = torch.nn.MSELoss()
epochs = configuration['Epochs']

# %% 
####################################
#Training Loop 
####################################
start_time = default_timer()
ll = []
for ep in range(epochs):

    model.train()
    train_loss, test_loss = 0, 0
    
    t1 = default_timer()
    for xx, yy in train_loader:
        optimizer.zero_grad()  # Add this line
        xx,yy = xx.to(device), yy.to(device)
        y_pred = model(xx)
        loss = loss_func(yy, y_pred)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()


    with torch.no_grad():
        for xx, yy in test_loader:
            xx,yy = xx.to(device), yy.to(device)
            y_pred = model(xx)
            loss = loss_func(yy, y_pred)
            test_loss += loss.item()

    t2 = default_timer()

    train_loss = train_loss / len(train_loader)
    test_loss = test_loss / len(test_loader)
    ll.append(train_loss)

    print(f"Epoch {ep}, Time Taken: {round(t2-t1,2)}, Train Loss: {round(train_loss, 5)}, Test Loss: {round(test_loss,5)}")
    
    scheduler.step()

train_time = default_timer() - start_time

plt.figure()
plt.plot(np.arange(epochs), ll)
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
# %%
# Saving the Model
saved_model = model_loc + case + '.pth'
torch.save(model.state_dict(), saved_model)

# %%  
#Validation 
# mse_loss = (out_test - model(in_test)).pow(2).mean()
mse_loss = (out_normalizer.decode(out_test) - out_normalizer.decode(model(in_test))).pow(2).mean()
print(f'MSE: {mse_loss}')
np.savez(os.getcwd() + '/Preds/' + case, targs=out_test, preds=model(in_test).detach())
# %%

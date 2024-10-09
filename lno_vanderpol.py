# Basado en el código original de qianyingcao (https://github.com/qianyingcao/Laplace-Neural-Operator)
# Licensed under the MIT License

'''
Author: Erik Jon Pérez Mardaras
email: erikjon.perez@gmail.com
LinkedIn: https://www.linkedin.com/in/erikjon-perez-mardaras/
Github: https://github.com/erikjonperez

September 2024
'''

#Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
import matplotlib.pyplot as plt
import os
import time
from timeit import default_timer
from utilities.utilities3 import *
from utilities.Adam import Adam
import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import scipy.io
import pandas as pd


#Check if CUDA is available
print(torch.cuda.is_available())

#Read the Data from the dataset
mat_file_path="./matlab_pocos_datos/datitos.mat"
mat_data = scipy.io.loadmat(mat_file_path)
dataframes = {}
for key in mat_data:
    if key.startswith('__'):
        continue
    dataframes[key] = pd.DataFrame(mat_data[key])

#Save Settings
folder_name = 'lno_model'
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
    print(f'Folder "{folder_name}" created.')
else:
    print(f'Folder "{folder_name}" already exists.')

save_index = 1   
current_directory = os.getcwd()
case = "Outputs"
folder_index = str(save_index)
results_dir = f"/lno_model/{case}{folder_index}/"
save_results_to = current_directory + results_dir
if not os.path.exists(save_results_to):
    os.makedirs(save_results_to)
    print(f'Folder "{results_dir}" created within "model".')
else:
    print(f'Folder "{results_dir}" already exists.')

#Classes defined
class PR(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(PR, self).__init__()

        self.modes1 = modes1
        self.scale = (1 / (in_channels*out_channels))
        self.weights_pole = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
        self.weights_residue = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))
       
    def output_PR(self, lambda1,alpha, weights_pole, weights_residue):   
        Hw=torch.zeros(weights_residue.shape[0],weights_residue.shape[0],weights_residue.shape[2],lambda1.shape[0], device=alpha.device, dtype=torch.cfloat)
        term1=torch.div(1,torch.sub(lambda1,weights_pole))
        Hw=weights_residue*term1
        output_residue1=torch.einsum("bix,xiok->box", alpha, Hw) 
        output_residue2=torch.einsum("bix,xiok->bok", alpha, -Hw) 
        return output_residue1,output_residue2    

    def forward(self, x):
        t=grid_x_train.cuda()
        #Compute input poles and resudes by FFT
        dt=(t[1]-t[0]).item()
        alpha = torch.fft.fft(x)
        lambda0=torch.fft.fftfreq(t.shape[0], dt)*2*np.pi*1j
        lambda1=lambda0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        lambda1=lambda1.cuda()
    
        # Obtain output poles and residues for transient part and steady-state part
        output_residue1,output_residue2= self.output_PR(lambda1, alpha, self.weights_pole, self.weights_residue)
    
        # Obtain time histories of transient response and steady-state response
        x1 = torch.fft.ifft(output_residue1, n=x.size(-1))
        x1 = torch.real(x1)
        x2=torch.zeros(output_residue2.shape[0],output_residue2.shape[1],t.shape[0], device=alpha.device, dtype=torch.cfloat)    
        term1=torch.einsum("bix,kz->bixz", self.weights_pole, t.type(torch.complex64).reshape(1,-1))
        term2=torch.exp(term1) 
        x2=torch.einsum("bix,ioxz->boz", output_residue2,term2)
        x2=torch.real(x2)
        x2=x2/x.size(-1)
        return x1+x2

class LNO1d(nn.Module):
    def __init__(self, width,modes):
        super(LNO1d, self).__init__()

        self.width = width
        self.modes1 = modes
        self.fc0 = nn.Linear(1, self.width) 

        self.conv0 = PR(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self,x):
        #grid = self.get_grid(x.shape, x.device)
        #x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 +x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x =  torch.sin(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
    
#Define parameters and load data
s = 2048  
batch_size_train = 20
batch_size_vali = 20
learning_rate = 0.000005
epochs = 1000
step_size = 100
gamma = 0.5
modes = 32 


reader = MatReader(mat_file_path)
x_train = reader.read_field('f_train') 
y_train = reader.read_field('u_train') 
grid_x_train = reader.read_field('x_train') 
x_vali = reader.read_field('f_vali')
y_vali = reader.read_field('u_vali')
grid_x_vali = reader.read_field('x_vali')
x_test = reader.read_field('f_test')
y_test = reader.read_field('u_test')
grid_x_test = reader.read_field('x_test') 

x_train = x_train.reshape(x_train.shape[0],s,1)
x_vali = x_vali.reshape(x_vali.shape[0],s,1) #redimensionameitnod e manera similar
x_test = x_test.reshape(x_test.shape[0],s,1) #rdimensionameinto de manera similar

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size_train, shuffle=True)

vali_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_vali, y_vali), batch_size=batch_size_vali, shuffle=True)

model = LNO1d(width,modes).cuda()

#Training-------------------------------------------------
mse_loss_fn = nn.L1Loss()
# = nn.MSELoss()
# = nn.L1Loss()
# = nn.SmoothL1Loss()
# = nn.BCELoss()
# = nn.BCEWithLogitsLoss()
# = nn.CrossEntropyLoss()
# = nn.CosineEmbeddingLoss()
# = nn.KLDivLoss()

myloss="nn.L1Loss()"

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma, last_epoch=-1, verbose='deprecated')
torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1, verbose='deprecated')
start_time = time.time()

train_error = np.zeros((epochs, 1))
train_loss = np.zeros((epochs, 1))
vali_error = np.zeros((epochs, 1))
vali_loss = np.zeros((epochs, 1))

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    train_l2 = 0
    n_train = 0
    for x, y in train_loader:
        x, y = x.cuda(), y.cuda()
        t = grid_x_train.cuda()
        optimizer.zero_grad()
        out = model(x)

        current_batch_size = x.size(0)

        mse = mse_loss_fn(out.view(current_batch_size, -1), y.view(current_batch_size, -1))
        mse.backward()
        optimizer.step()
        train_mse += mse.item()
        n_train += 1

        del x, y, out, mse
        torch.cuda.empty_cache()

    scheduler.step()
    model.eval()
    vali_mse = 0.0
    with torch.no_grad():
        n_vali = 0
        for x, y in vali_loader:
            x, y = x.cuda(), y.cuda()
            t = grid_x_vali.cuda()
            out = model(x)

            current_batch_size = x.size(0)
            mse = mse_loss_fn(out.view(current_batch_size, -1), y.view(current_batch_size, -1))

            vali_mse += mse.item()
            n_vali += 1

    train_mse /= n_train
    vali_mse /= n_vali
    train_loss[ep, 0] = train_mse
    vali_loss[ep, 0] = vali_mse
    t2 = default_timer()
    print("Epoch: %d, time: %.3f, Train Loss: %.3e, Vali Loss: %.3e" % (ep, t2 - t1, train_mse, vali_mse))

elapsed = time.time() - start_time
print("\n=============================")
print("Training done...")
print('Training time: %.3f' % (elapsed))
print("=============================\n")


#Saving settings-----------------------------------------
x = np.linspace(0, epochs-1, epochs)
np.savetxt(save_results_to+'/epoch.txt', x)
np.savetxt(save_results_to+'/train_loss.txt', train_loss)
np.savetxt(save_results_to+'/vali_loss.txt', vali_loss)
np.savetxt(save_results_to+'/train_error.txt', train_error)
np.savetxt(save_results_to+'/vali_error.txt', vali_error)    
save_models_to = save_results_to +"model/"
if not os.path.exists(save_models_to):
    os.makedirs(save_models_to)
    
path_guardado='./lno_model/pesitos.pth'
torch.save(model.state_dict(), path_guardado) 
print("Modelo guardado en: ", path_guardado)


#testing---------------
def load_data(file_path):
    return np.loadtxt(file_path)
epochs = load_data('./lno_model/Outputs1/epoch.txt')         
train_loss = load_data('./lno_model/Outputs1/train_loss.txt')
vali_loss = load_data('./lno_model/Outputs1/vali_loss.txt')
print(f"epochs shape: {epochs.shape}")
print(f"train_loss shape: {train_loss.shape}")
print(f"vali_loss shape: {vali_loss.shape}")
fig = plt.figure(constrained_layout=False, figsize=(7, 7))
gs = fig.add_gridspec(1, 1)
ax = fig.add_subplot(gs[0])
ax.plot(epochs, train_loss, color='blue', label='Train Loss')
ax.plot(epochs, vali_loss, color='red', label='Validation Loss')
ax.set_yscale('log') 
ax.set_ylabel('Loss')
ax.set_xlabel('Epochs')
ax.legend(loc='upper left')
save_directory = './lno_model/Outputs1/'  
if not os.path.exists(save_directory):
    os.makedirs(save_directory)  
fig.savefig(os.path.join(save_directory, 'loss_history.png'))
plt.show()

#create hiperparameter file-----
directory = './lno_model/'  
file_name = 'Hiperparametros.txt'
file_path = os.path.join(directory, file_name)
os.makedirs(directory, exist_ok=True)
with open(file_path, 'w') as file:
    file.write(f'Discretization s: {s}\n')
    file.write(f'batch_size train: {batch_size_train}\n')
    file.write(f'batch_size vali: {batch_size_vali}\n')
    file.write(f'learning_rate: {learning_rate}\n')
    file.write(f'epochs: {epochs}\n')
    file.write(f'step_size: {step_size}\n')
    file.write(f'gamma: {gamma}\n')
    file.write(f'modes: {modes}\n')
    file.write(f'width: {width}\n')
    file.write(f'optimizer: {optimizer}\n')
    file.write(f'scheduler: {scheduler}\n')
    file.write(f'loss: {myloss}\n')
print(f'Datos guardados en {file_path}')
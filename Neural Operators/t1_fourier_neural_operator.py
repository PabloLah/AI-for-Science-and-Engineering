import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt


def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['lrelu', 'LReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus(beta=4)
    elif name in ['celu', 'CeLU']:
        return nn.CELU()
    elif name in ['elu']:
        return nn.ELU()
    elif name in ['mish']:
        return nn.Mish()
    else:
        raise ValueError('Unknown activation function')

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1, device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        # Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x


class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.padding = 7  # pad the domain if input is non-periodic
        self.linear_p = nn.Linear(2, self.width)  # input channel is 2: (u0(x), x) --> GRID IS INCLUDED!

        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.lin0 = nn.Conv1d(self.width, self.width, 1)
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)

        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 1)

        self.activation = torch.nn.Tanh()

    def fourier_layer(self, x, spectral_layer, conv_layer):
        return self.activation(spectral_layer(x) + conv_layer(x))

    def linear_layer(self, x, linear_transformation):
        return self.activation(linear_transformation(x))

    def forward(self, x):
        x = self.linear_p(x)
        x = x.permute(0, 2, 1)

        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        x = self.fourier_layer(x, self.spect1, self.lin0)
        x = self.fourier_layer(x, self.spect2, self.lin1)
        x = self.fourier_layer(x, self.spect3, self.lin2)

        x = x[..., :-self.padding]  # pad the domain if input is non-periodic
        x = x.permute(0, 2, 1)

        x = self.linear_layer(x, self.linear_q)
        x = self.output_layer(x)
        return x

torch.manual_seed(0)
np.random.seed(0)


# Load the data
#read data from files
data_numpy = np.loadtxt('project_2/data/TrainingData.txt', delimiter=',', skiprows=1)
predict_data_numpy = np.loadtxt('project_2/data/TestingData.txt', delimiter=',', skiprows=1)
#convert into torch tensor
tensor_data = torch.tensor(data_numpy, dtype=torch.float32)
tensor_predict_times = torch.tensor(predict_data_numpy, dtype=torch.float32)
#normalize the time column in the tensors
predict_times_normalized = (tensor_predict_times - tensor_data[:,0].min(0, keepdim=True)[0]) / (tensor_predict_times.max(0, keepdim=True)[0] - tensor_data[:,0].min(0, keepdim=True)[0])
tensor_data[:,0] = (tensor_data[:,0] - tensor_data[:,0].min(0, keepdim=True)[0]) / (tensor_predict_times.max(0, keepdim=True)[0] - tensor_data[:,0].min(0, keepdim=True)[0])
#instead normalize wrt to the general maximum temperature; better evaluation results then col-wise
tensor_data[:,1:3] = tensor_data[:,1:3] / tensor_data[:,1:3].max()

#split data into Tf and Ts
samples_train_Tf = tensor_data[:,[1, 0]] # since input should be of the form (Tf(t), t) i.e. (Ts(t), t)
samples_train_Ts = tensor_data[:,[2, 0]]

#aggregate samples into windows of size n
n=34
input_train_Tf = samples_train_Tf.unfold(0, n, 1)[:-n-1].permute(0, 2, 1) 
output_train_Tf = samples_train_Tf[:, 0].unfold(0, n, 1)[1:-n] #since output should only include Tf values

input_train_Ts = samples_train_Ts.unfold(0, n, 1)[:-n-1].permute(0, 2, 1)
output_train_Ts = samples_train_Ts[:, 0].unfold(0, n, 1)[1:-n] #since output should only include Ts values

PERFORM_TRAINIG = True
if PERFORM_TRAINIG: 
    batch_size = 10
    #dataloader
    train_dataloader = DataLoader(TensorDataset(input_train_Tf, output_train_Tf, input_train_Ts, output_train_Ts), batch_size=batch_size, shuffle=False)
    #hyperparams
    learning_rate = 0.005
    epochs = 50
    step_size = 50
    gamma = 0.5
    #best so far: lr = 0.005, epochs=50

    modes = 16
    width = 64

    #define models for Tf and Ts
    fno_Tf = FNO1d(modes, width) 
    fno_Ts = FNO1d(modes, width) 


    parameters = list(fno_Tf.parameters()) + list(fno_Ts.parameters())
    optimizer = Adam(parameters, lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    #TRAIN
    optimizer_Tf = Adam(fno_Tf.parameters(), lr=learning_rate, weight_decay=1e-5)
    optimizer_Ts = Adam(fno_Ts.parameters(), lr=learning_rate, weight_decay=1e-5)

    scheduler_Tf = torch.optim.lr_scheduler.StepLR(optimizer_Tf, step_size=step_size, gamma=gamma)
    scheduler_Ts = torch.optim.lr_scheduler.StepLR(optimizer_Ts, step_size=step_size, gamma=gamma)

    l = torch.nn.MSELoss()
    freq_print = 1
    for epoch in range(epochs):
        train_mse_Tf = 0.0
        train_mse_Ts = 0.0

        for step, (input_Tf, output_Tf, input_Ts, output_Ts) in enumerate(train_dataloader):
            optimizer_Tf.zero_grad()
            optimizer_Ts.zero_grad()

            output_pred_batch_Tf = fno_Tf(input_Tf).squeeze(2)
            output_pred_batch_Ts = fno_Ts(input_Ts).squeeze(2)
            loss_Tf = l(output_pred_batch_Tf, output_Tf)
            loss_Ts = l(output_pred_batch_Ts, output_Ts)

            loss_Tf.backward()
            loss_Ts.backward()

            optimizer_Tf.step()
            optimizer_Ts.step()

            train_mse_Tf += loss_Tf.item()
            train_mse_Ts += loss_Ts.item()

        train_mse_Tf /= len(train_dataloader)
        train_mse_Ts /= len(train_dataloader)


        scheduler_Tf.step()
        scheduler_Ts.step()

        if epoch % freq_print == 0: print("######### Epoch:", epoch, " ######### Train Loss Tf:", train_mse_Tf, " ######### Train Loss Ts:", train_mse_Ts)

# save and load models
SAVE_MODEL = False
if SAVE_MODEL and PERFORM_TRAINIG:
    torch.save(fno_Tf, "project_2/models_task1/fno_Tf_model.pt")

    torch.save(fno_Ts, "project_2/models_task1/fno_Ts_model.pt")
    print("model was saved!")

LOAD_MODEL = False
if LOAD_MODEL:
    fno_Tf = torch.load("project_2/models_task1/fno_Tf_model.pt")
    fno_Ts = torch.load("project_2/models_task1/fno_Ts_model.pt")
    print("model was loaded!")

# PREDICT
#as first set for prediction use last window of length 34 of training set. 
# then continuously use the data from the predictions
#prepare model input
last_training_window_Tf = samples_train_Tf[-n:, 0] #extract temp values
last_training_window_Ts = samples_train_Ts[-n:, 0]
t_col = samples_train_Tf[-n:, 1].tolist() #list for later easily appending the new t value

with torch.no_grad():
    fno_Tf.eval()
    fno_Ts.eval()

    #adapt input to correct input dimensions    
    pred_Tf = last_training_window_Tf.unsqueeze(0).unsqueeze(2)
    pred_Ts = last_training_window_Ts.unsqueeze(0).unsqueeze(2) 

    predicted_t_Tf_vals = []
    predicted_t_Ts_vals = []
    for t_new in predict_times_normalized.tolist():
        t_col_tensor = torch.tensor(t_col, dtype=torch.float32).unsqueeze(0).unsqueeze(2)
        input_Tf = torch.cat((pred_Tf, t_col_tensor), dim=2)
        input_Ts = torch.cat((pred_Ts, t_col_tensor), dim=2)
        pred_Tf = fno_Tf(input_Tf)
        pred_Ts = fno_Ts(input_Ts)

        #only append the predicted datapoint
        predicted_t_Tf_vals.append(pred_Tf[0, -1, 0].item())
        predicted_t_Ts_vals.append(pred_Ts[0, -1, 0].item())
        t_col = t_col[1:]
        t_col.append(t_new)

Tf_values = samples_train_Tf[:, 0].tolist() + predicted_t_Tf_vals
Ts_values = samples_train_Ts[:, 0].tolist() + predicted_t_Ts_vals
time_values = samples_train_Tf[:, 1].tolist() + predict_times_normalized.tolist()
plt.plot(time_values, Tf_values, label='Tf values')
plt.plot(time_values, Ts_values, label='Ts values')
plt.axvline(x=predict_times_normalized.tolist()[0], color='red', linestyle='--', label='first preticted time')
plt.axvline(x=samples_train_Tf[:, 1].tolist()[-n], color='green', linestyle='--', label='last time seen in training')
plt.legend()
plt.show()

WRITE_DATA_FILE = False
if WRITE_DATA_FILE:
    #rescale temp values
    predicted_t_Tf_vals = np.array(predicted_t_Tf_vals) * tensor_data[:,1:3].max().item()
    predicted_t_Ts_vals = np.array(predicted_t_Ts_vals) * tensor_data[:,1:3].max().item()
    #predict_data_numpy = testingData which was read into the file at the beginning of the code.
    data_to_write = np.column_stack((predict_data_numpy, predicted_t_Tf_vals, np.array(predicted_t_Ts_vals)))
    #write file for submission
    file_path = "Task1_submission.txt"
    # Write data to text file
    with open(file_path, "w") as file:
        # Write the header
        file.write("t,tf0,ts0\n")
        # Write the data
        for row in data_to_write:
            file.write(",".join(map(str, row)) + "\n")
    print("Prediction file was written!")

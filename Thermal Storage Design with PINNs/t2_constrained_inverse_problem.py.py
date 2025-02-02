import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(342)

#define dense NN class
class NNAnsatz(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NNAnsatz, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


#Define Network Class
class Pinns:
    def __init__(self, n_int_, n_sb_, n_tb_, n_supervised_):
        self.n_int = n_int_ #how many interior training points do we want?
        self.n_sb = n_sb_ #how many spatial boundary points
        self.n_tb = n_tb_ #how many temporal boundary training points to we want?
        self.n_supervised = n_supervised_ # max 39.201

        # Extrema of the solution domain (x,t) in [0,1]x[0,8]
        self.domain_extrema = torch.tensor([[0.0, 1.0], # Space dimension
                                            [0.0, 8.0]])  # Time dimension

        # Constants
        self.alpha_f = 0.005
        self.h_f = 5.0
        self.T_hot = 4.0
        self.T_cold = 1.0
        self.T_0 = 1.0

        # Parameter to balance role of data and PDE
        self.lambda_boundary = 1.0
        self.zeta_interior = 1.0
        self.gamma_supervised = 15.0

        # Dense NN to approximate the NN for Ts and Ts (see Ts as coefficient of the inverse problem)
        # Tf, Ts
        self.approximate_T = NNAnsatz(input_dim = 2, hidden_dim = 32, output_dim = 2)
                
        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        assert(self.domain_extrema.shape[0] == 2)

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb_0, self.training_set_sb_1, self.training_set_tb, self.training_set_int = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    # Initial condition to solve, i.e. the supposed initial value of Tf (constant T_0)
    def initial_condition(self, x):
        return torch.full((x.shape[0], 1), self.T_0) #this should be a 1-dim array, for the output of Tf

    ################################################################################################
    # Generate training data 
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[1, 0] #check that this outputs the correct value
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 1] = torch.full(input_tb[:, 1].shape, t0) #replace time values with constant t0 value (i.e. second col of input)
        output_tb = self.initial_condition(input_tb[:, 0]).reshape(-1, 1)#.repeat(1,2)
        return input_tb, output_tb

    # spatial boundary without derivatives
    #add boundary points corresponding to the spatial boundary conditions (i.e. points at 0 and at 1)
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[0, 0]
        x1 = self.domain_extrema[0, 1]

        input_sb = self.convert(self.soboleng.draw(self.n_sb))
        input_sb_0 = torch.clone(input_sb) #is this cloning really necessary? 
        input_sb_0[:, 0] = torch.full(input_sb[:, 0].shape, x0).requires_grad_(True)
        input_sb_1 = torch.clone(input_sb) #is this cloning really necessary? 
        input_sb_1[:, 0] = torch.full(input_sb[:, 0].shape, x1).requires_grad_(True)

        output_sb_0 = torch.zeros((input_sb_1.shape[0], 1)).requires_grad_(True)
        output_sb_1 = torch.zeros((input_sb_1.shape[0], 1)).requires_grad_(True)
        
        return input_sb_0, output_sb_0, input_sb_1, output_sb_1

    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int
    
    # Function to get measured data in txt file
    def get_measured_data(self):
        #d = pd.read_csv('DataSolution.txt')
        #input = torch.tensor(d[['t', 'x']].values, dtype=torch.float32)
        tensor_data = np.loadtxt('DataSolution.txt', delimiter=',', skiprows=1)
        num_rows = len(tensor_data)
        random_indices = torch.randperm(num_rows)[:self.n_supervised]
        random_rows_data = tensor_data[random_indices]
        tensor_data = torch.tensor(random_rows_data, dtype=torch.float32)
        #tensor_data = torch.tensor(tensor_data, dtype=torch.float32)
        #debugging
        #self.plotting_for_testing(tensor_data)

        #dont forget to flip the order of x,t (data order: t,x; code order: x,t)
        return tensor_data[:, [1, 0]], tensor_data[:,2]
    
    #---------DEBUGGING--------------
    # Compute the histogram
    def histogram_test(self, column_data):
        histogram = torch.histc(column_data, bins=10, min=column_data.min(), max=column_data.max())

        # Plot the histogram
        plt.bar(range(len(histogram)), histogram)
        plt.xlabel('Bin')
        plt.ylabel('Frequency')
        plt.title('Histogram of Column Data')
        plt.show()
        return
    
    def plotting_for_testing(self, inputs):
        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 0], inputs[:, 1], c=inputs[:, 2], cmap="jet", s=5)
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")

        im2 = axs[1].scatter(inputs[:, 0], inputs[:, 1], c=inputs[:, 2], cmap="jet", s=5)
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Tf (fluid temperature)")
        axs[1].set_title("Ts (solid temperature)")

        plt.show()
        return
    #--------------------------------

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb_0, output_sb_0, input_sb_1, output_sb_1 = self.add_spatial_boundary_points()   # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()         # S_int

        training_set_sb_0 = DataLoader(torch.utils.data.TensorDataset(input_sb_0, output_sb_0), batch_size=self.n_sb, shuffle=False)
        training_set_sb_1 = DataLoader(torch.utils.data.TensorDataset(input_sb_1, output_sb_1), batch_size=self.n_sb, shuffle=False)
        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)
        #training_set_supervised = DataLoader(torch.utils.data.TensorDataset(input_supervised, output_supervised), batch_size=self.n_supervised, shuffle=False)


        return training_set_sb_0, training_set_sb_1, training_set_tb, training_set_int#, training_set_supervised

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    #is this function really necessary??
    def apply_initial_condition(self, input_tb):
        #approximated values of Tf
        u_pred_tb = self.approximate_T(input_tb)[:, 0].reshape(-1, 1) #NN approximation of values
        return u_pred_tb

    # Function to find if a t-value belongs to a certain phase (charge, discharge, idle)
    # for an array to t values [0,8] it returns a bool array of whether the t value is in the speficied phase
    def time_is_phase(self, t, phase):
        #t is array of time values
        if phase == 'charge':
            is_phase_bool = ( (t >= 0.0) & (t <= 1.0) ) | ( (t >= 4.0) & (t <= 5.0) )
        elif phase == 'discharge':
            is_phase_bool = ( (t >= 2.0) & (t <= 3.0) ) | ( (t >= 6.0) & (t <= 7.0) )
        elif phase == 'idle':
            is_phase_bool =  (( (t >= 1.0) & (t <= 2.0) ) | ( (t >= 3.0) & (t <= 4.0) ) | ( (t >= 5.0) & (t <= 6.0) ) | ( (t >= 7.0) & (t <= 8.0) ))
        else: raise ValueError("invalid phase")
        return is_phase_bool

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    # return terms for derivatives of Tf and Ts, use torch gradient
    def apply_boundary_conditions(self, input_sb_0, input_sb_1):
        Tf_pred_sb_0 = self.approximate_T(input_sb_0)[:, 0].reshape(-1, 1)
        Tf_pred_sb_1 = self.approximate_T(input_sb_1)[:, 0].reshape(-1, 1)

        #partition points into their respective states (charge, discharge, idle), and then compute the gradients
        #this prevents us from computing unncecssary derivatives
        # bool vector giving information about the rows which are relevant for each phase        
        charge_bool_0 = self.time_is_phase(input_sb_0[:, 1], 'charge')
        charge_bool_1 = self.time_is_phase(input_sb_1[:, 1], 'charge')
        discharge_bool_0 = self.time_is_phase(input_sb_0[:, 1], 'discharge')
        discharge_bool_1 = self.time_is_phase(input_sb_1[:, 1], 'discharge')
        idle_bool_0 = self.time_is_phase(input_sb_0[:, 1], 'idle')
        idle_bool_1 = self.time_is_phase(input_sb_1[:, 1], 'idle')

        # compute relevant boundary values for each phase
        # compute gradients
        Tf_x_0 = torch.autograd.grad(Tf_pred_sb_0.sum(), input_sb_0, create_graph=True, allow_unused=True)[0][:, 0] #derivative wrt x = input_sb_1[:, 0]
        Tf_x_1 = torch.autograd.grad(Tf_pred_sb_1.sum(), input_sb_1, create_graph=True, allow_unused=True)[0][:, 0]

        #now partition gradients into ther resp. phases
        #charge phase: 
        Tf_charge_0 = Tf_pred_sb_0[charge_bool_0]
        Tf_x_charge_1 =  Tf_x_1[charge_bool_1]
        # discharge phase
        Tf_x_discharge_0 = Tf_x_0[discharge_bool_0]
        Tf_discharge_1 = Tf_pred_sb_1[discharge_bool_1]
        #idle phase
        Tf_x_idle_0 = Tf_x_0[idle_bool_0]
        Tf_x_idle_1 = Tf_x_1[idle_bool_1]

        return Tf_charge_0, Tf_x_charge_1, Tf_x_discharge_0, Tf_discharge_1, Tf_x_idle_0, Tf_x_idle_1
    
    def apply_supervised_points(self, input_supervised):
        pred_Tf_supervised = self.approximate_T(input_supervised)[:, 0].reshape(-1, 1)
        return pred_Tf_supervised
    
    def Uf_fluid_velocity(self, t):
        #get the bool vectors, indication which phase each row is in
        charge_bool = self.time_is_phase(t, 'charge')
        discharge_bool = self.time_is_phase(t, 'discharge')
        idle_bool = self.time_is_phase(t, 'idle')

        #compute vector with Uf values (1 when charge, 0 when idle, -1 when discharge)
        #problem with overlapping points
        Uf = 1 * charge_bool + (-1) * discharge_bool + 0 * idle_bool
        return Uf

    # Function to compute the PDE residuals
    def compute_pde_residuals(self, input_int):
        input_int.requires_grad = True
        T = self.approximate_T(input_int)
        Tf = T[:, 0].reshape(-1, 1)
        Ts = T[:, 1].reshape(-1, 1)

        Uf = self.Uf_fluid_velocity(input_int[:, 1]).reshape(-1, 1)

        Tf_grad = torch.autograd.grad(Tf.sum(), input_int, create_graph=True)[0]

        Tf_x = Tf_grad[:, 0].reshape(-1, 1)
        Tf_t = Tf_grad[:, 1].reshape(-1, 1)

        Tf_xx = torch.autograd.grad(Tf_x.sum(), input_int, create_graph=True, allow_unused=True)[0][:, 0].reshape(-1, 1)

        residual = Tf_t + Uf * Tf_x - self.alpha_f * Tf_xx + self.h_f * (Tf - Ts)#(Tf.view(-1) - Ts.view(-1))
        
        return residual.reshape(-1, ) # is reshape really necessary?

    #delete unused variables from whole code
    def compute_loss(self, inp_train_sb_0, u_train_sb_0, inp_train_sb_1, u_train_sb_1, inp_train_tb, u_train_tb, inp_train_int, input_supervised, Tf_supervised, verbose=True):
        
        # get supervised points
        print("shape of supervised input during loss computation: ", input_supervised.shape)

        #compute predictions of spatial, temporal boundary, and of supervised points
        pred_Tf_charge_0, pred_Tf_x_charge_1, pred_Tf_x_discharge_0, pred_Tf_discharge_1, pred_Tf_x_idle_0, pred_Tf_x_idle_1 = self.apply_boundary_conditions(inp_train_sb_0, inp_train_sb_1)
        pred_Tf_tb = self.apply_initial_condition(inp_train_tb)
        pred_Tf_supervised = self.apply_supervised_points(input_supervised)

        # compute the residuals
        # interior residual
        residual_int = self.compute_pde_residuals(inp_train_int)

        # spatial boundary condition residuals (should all equal zero)
        residual_sb_charge_0 = pred_Tf_charge_0 - self.T_hot
        residual_sb_charge_1 = pred_Tf_x_charge_1
        residual_sb_discharge_0 = pred_Tf_x_discharge_0
        residual_sb_discharge_1 = pred_Tf_discharge_1 - self.T_cold
        residual_sb_idle_0 = pred_Tf_x_idle_0
        residual_sb_idle_1 = pred_Tf_x_idle_1

        # temporal boundary
        residual_tb = pred_Tf_tb - self.T_0

        #supervised residual
        residual_supervised = Tf_supervised - pred_Tf_supervised

        #compute losses
        loss_interior = torch.mean(abs(residual_int) ** 2)

        loss_sb_charge_0 = torch.mean(abs(residual_sb_charge_0) ** 2)
        loss_sb_charge_1 = torch.mean(abs(residual_sb_charge_1) ** 2)
        loss_sb_discharge_0 = torch.mean(abs(residual_sb_discharge_0) ** 2)
        loss_sb_discharge_1 = torch.mean(abs(residual_sb_discharge_1) ** 2)
        loss_sb_idle_0 = torch.mean(abs(residual_sb_idle_0) ** 2)
        loss_sb_idle_1 = torch.mean(abs(residual_sb_idle_1) ** 2)

        loss_tb = torch.mean(abs(residual_tb) ** 2)

        loss_supervised = torch.mean(abs(residual_supervised) ** 2)

        #group boundary losses
        #loss_interior
        loss_boundary = loss_tb + loss_sb_charge_0 + loss_sb_charge_1 + loss_sb_discharge_0 + loss_sb_discharge_1 + loss_sb_idle_0 + loss_sb_idle_1

        loss = torch.log10(self.zeta_interior * loss_interior + self.lambda_boundary * loss_boundary + self.gamma_supervised * loss_supervised)
        
        if verbose: print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(loss_boundary).item(), 4), "| Function Loss: ", round(torch.log10(loss_interior).item(), 4), "| Supervised Loss: ", round(torch.log10(loss_supervised).item(), 4))

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()
        #print("shape of supervised input during fitting: ", input_supervised.shape)

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((inp_train_sb_0, u_train_sb_0), (inp_train_sb_1, u_train_sb_1), (inp_train_tb, u_train_tb), (inp_train_int, u_train_int)) in enumerate(zip(self.training_set_sb_0, self.training_set_sb_1 , self.training_set_tb, self.training_set_int)):
                #redraw the supervised points at every run; 
                input_supervised, output_supervised = self.get_measured_data() #data for supervised training

                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb_0, u_train_sb_0, inp_train_sb_1, u_train_sb_1, inp_train_tb, u_train_tb, inp_train_int, input_supervised, output_supervised, verbose=verbose)
                    loss.backward(retain_graph=True)

                    history.append(loss.item())
                    return loss

                optimizer.step(closure=closure)

        print('Final Loss: ', history[-1])

        return history

    ################################################################################################
    def plotting(self):
        inputs = self.soboleng.draw(100000)
        inputs = self.convert(inputs)
        print(inputs.shape)
        output_Tf = self.approximate_T(inputs)[:, 0].reshape(-1, 1)
        output_Ts = self.approximate_T(inputs)[:, 1].reshape(-1, 1)

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_Tf.detach(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")

        im2 = axs[1].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output_Ts.detach(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Tf (fluid temperature)")
        axs[1].set_title("Ts (solid temperature)")

        plt.show()

# perform the training
multiple = 2
n_int = 128 * multiple
n_sb = 500 * multiple
n_tb = 64 * multiple
n_supervised = 3000 * multiple

pinn = Pinns(n_int, n_sb, n_tb, n_supervised)

max_iter = 10000
n_epochs = 5
optimizer_LBFGS = optim.LBFGS(pinn.approximate_T.parameters(),
                              lr=float(0.5),
                              max_iter=max_iter,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)
optimizer_ADAM = optim.Adam(pinn.approximate_T.parameters(),
                            lr=float(0.005))


hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)

def evaluate_task_data(model):
    tensor_data = np.loadtxt('SubExample.txt', delimiter=',', skiprows=1)
    tensor_data = torch.tensor(tensor_data, dtype=torch.float32)
    # format: t,x,ts

    with torch.no_grad():  # Disable gradient tracking since we're only predicting
        #switch order of t,x to x,t
        output = model(tensor_data[:, [1, 0]])[:, 1].reshape(-1, 1) # only read out values of Ts
    #print("max of Ts values: ", torch.max(output))
    # write predictions into file
    # Convert tensor data to list of lists
        #only take t,x values from tensor_data
    predictions = torch.cat((tensor_data[:, [0, 1]], output), dim=1)

    list_data = predictions.tolist()

    # Define the file path
    file_path = "submission.txt"

    # Write data to text file
    with open(file_path, "w") as file:
        # Write the header
        file.write("t,x,ts\n")
        
        # Write the data
        for row in list_data:
            file.write(",".join(map(str, row)) + "\n")

evaluate_task_data(pinn.approximate_T)
print("file has been written")
    

pinn.plotting()
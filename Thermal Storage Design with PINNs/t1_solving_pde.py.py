import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
torch.autograd.set_detect_anomaly(True)
torch.manual_seed(128)

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
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.layers(x)


#Define Network Class
class Pinns:
    def __init__(self, n_int_, n_sb_, n_tb_):
        self.n_int = n_int_ #how many interior training points do we want?
        self.n_sb = n_sb_ #how many spatial boundary points
        self.n_tb = n_tb_ #how many temporal boundary training points to we want?

        # Extrema of the solution domain (x,t) in [0,1]x[0,1]
        self.domain_extrema = torch.tensor([[0.0, 1.0], # Space dimension
                                            [0.0, 1.0]])  # Time dimension

        # Constants
        self.alpha_f = 0.05
        self.alpha_s = 0.08
        self.h_f = 5.0
        self.h_s = 6.0
        self.T_hot = 4.0
        self.T_0 = 1.0
        self.U_f = 1.0

        # Parameter to balance role of data and PDE
        self.lambda_u = 10

        # F Dense NN to approximate the solution of the underlying heat equation
        self.approximate_solution = NNAnsatz(input_dim = 2, hidden_dim = 20, output_dim = 2)
                                            
        # Generator of Sobol sequences
        self.soboleng = torch.quasirandom.SobolEngine(dimension=self.domain_extrema.shape[0])
        assert(self.domain_extrema.shape[0] == 2)

        # Training sets S_sb, S_tb, S_int as torch dataloader
        self.training_set_sb_0, self.training_set_sb_1 , self.training_set_tb, self.training_set_int = self.assemble_datasets()

    ################################################################################################
    # Function to linearly transform a tensor whose value are between 0 and 1
    # to a tensor whose values are between the domain extrema
    def convert(self, tens):
        assert (tens.shape[1] == self.domain_extrema.shape[0])
        return tens * (self.domain_extrema[:, 1] - self.domain_extrema[:, 0]) + self.domain_extrema[:, 0]

    # Initial condition to solve, i.e. the supposed initial value of Tf and Ts (constant T_0)
    def initial_condition(self, x):
        return torch.full((x.shape[0], 2), self.T_0)

    def spatial_boundary_condition(self, t):
        return (self.T_hot - self.T_0)/(1+ torch.exp(- 200 * (t - 0.25))) + self.T_0
    

    ################################################################################################
    # Generate training data 
    def add_temporal_boundary_points(self):
        t0 = self.domain_extrema[1, 0] #check that this outputs the correct value
        input_tb = self.convert(self.soboleng.draw(self.n_tb))
        input_tb[:, 1] = torch.full(input_tb[:, 0].shape, t0) #replace time values with constant t0 value (i.e. second col of input)
        output_tb = self.initial_condition(input_tb[:, 0])#.reshape(-1, 1).repeat(1,2)

        return input_tb, output_tb

    #add boundary points corresponding to the spatial boundary conditions 
    def add_spatial_boundary_points(self):
        x0 = self.domain_extrema[0, 0]
        x1 = self.domain_extrema[0, 1]

        #simplify by drawing only once and putting it into variable imput_sb, then cloning this tensor to get sb1 and sb0
        input_sb_0 = self.convert(self.soboleng.draw(self.n_sb))
        input_sb_1 = self.convert(self.soboleng.draw(self.n_sb))

        input_sb_0 = torch.clone(input_sb_0) #is this cloning really necessary? Good practice?
        input_sb_0[:, 0] = torch.full(input_sb_0[:, 0].shape, x0).requires_grad_(True)
        input_sb_1 = torch.clone(input_sb_1) #is this cloning really necessary? TEST!!
        input_sb_1[:, 0] = torch.full(input_sb_1[:, 0].shape, x1).requires_grad_(True)

        # view reshapes vector into a column vector of t-values
        spatial_boundary_values = self.spatial_boundary_condition(input_sb_0[:, 1]).clone().detach().view(-1, 1)

        # Concatenate tensors along the second dimension (columns)
        # values for Ts don't matter, so we simply duplicate them to be the values of Tf (thus two identical columns)
        ## DOES IT MATTER WHICH EXACT VALUES WE INCLUDE?? ##
        output_sb_0 = torch.cat((spatial_boundary_values, torch.zeros((input_sb_0.shape[0], 1))), dim=1).requires_grad_(True)
        output_sb_1 = torch.zeros((input_sb_1.shape[0], 2)).requires_grad_(True)
        
        return input_sb_0, output_sb_0, input_sb_1, output_sb_1

    #  Function returning the input-output tensor required to assemble the training set S_int corresponding to the interior domain where the PDE is enforced
    def add_interior_points(self):
        input_int = self.convert(self.soboleng.draw(self.n_int))
        output_int = torch.zeros((input_int.shape[0], 1))
        return input_int, output_int

    # Function returning the training sets S_sb, S_tb, S_int as dataloader
    def assemble_datasets(self):
        input_sb_0, output_sb_0, input_sb_1, output_sb_1 = self.add_spatial_boundary_points()   # S_sb
        input_tb, output_tb = self.add_temporal_boundary_points()  # S_tb
        input_int, output_int = self.add_interior_points()         # S_int

        training_set_sb_0 = DataLoader(torch.utils.data.TensorDataset(input_sb_0, output_sb_0), batch_size=self.n_sb, shuffle=False)
        training_set_sb_1 = DataLoader(torch.utils.data.TensorDataset(input_sb_1, output_sb_1), batch_size=self.n_sb, shuffle=False)

        training_set_tb = DataLoader(torch.utils.data.TensorDataset(input_tb, output_tb), batch_size=self.n_tb, shuffle=False)
        training_set_int = DataLoader(torch.utils.data.TensorDataset(input_int, output_int), batch_size=self.n_int, shuffle=False)

        return training_set_sb_0, training_set_sb_1, training_set_tb, training_set_int

    ################################################################################################
    # Function to compute the terms required in the definition of the TEMPORAL boundary residual
    def apply_initial_condition(self, input_tb):
        u_pred_tb = self.approximate_solution(input_tb) #NN approximation of values
        return u_pred_tb

    # Function to compute the terms required in the definition of the SPATIAL boundary residual
    # return terms for derivatives of Tf and Ts, use torch gradient
    def apply_boundary_conditions(self, input_sb_0, input_sb_1):
        T_pred_sb_0 = self.approximate_solution(input_sb_0)
        T_pred_sb_1 = self.approximate_solution(input_sb_1)

        #non neumann boundary condition
        Tf_0 = T_pred_sb_0[:, 0] #extract first col of Tf values
        # first residual 
        Ts_0 = T_pred_sb_0[:, 1]  #extract second column of Ts values
        grad_Ts_x_0 = torch.autograd.grad(Ts_0.sum(), input_sb_0, create_graph=True)[0][:, 0]
        # second residual
        Ts_1 = T_pred_sb_1[:, 1]  
        grad_Ts_x_1 = torch.autograd.grad(Ts_1.sum(), input_sb_1, create_graph=True)[0][:, 0]
        # third residual
        Tf_1 = T_pred_sb_1[:, 0]  
        grad_Tf_x_1 = torch.autograd.grad(Tf_1.sum(), input_sb_1, create_graph=True)[0][:, 0]
        return Tf_0, grad_Ts_x_0, grad_Ts_x_1, grad_Tf_x_1

    # Function to compute the PDE residuals
    def compute_pde_residuals(self, input_int):
        input_int.requires_grad = True
        output = self.approximate_solution(input_int)
        Tf = output[:, 0]  # Extracting the first output (Tf)
        Ts = output[:, 1]  # Extracting the second output (Ts)
    
        grad_Tf = torch.autograd.grad(Tf.sum(), input_int, create_graph=True)[0]
        grad_Ts = torch.autograd.grad(Ts.sum(), input_int, create_graph=True)[0]

        grad_Tf_x = grad_Tf[:, 0]
        grad_Tf_t = grad_Tf[:, 1]
        grad_Ts_x = grad_Ts[:, 0]
        grad_Ts_t = grad_Ts[:, 1]
        grad_Tf_xx = torch.autograd.grad(grad_Tf_x.sum(), input_int, create_graph=True)[0][:, 0]
        grad_Ts_xx = torch.autograd.grad(grad_Ts_x.sum(), input_int, create_graph=True)[0][:, 0]

        residual1 = grad_Tf_t + self.U_f * grad_Tf_x - self.alpha_f * grad_Tf_xx + self.h_f * (Tf - Ts)
        residual2 = grad_Ts_t - self.alpha_s * grad_Ts_xx - self.h_s * (Tf - Ts)
        return residual1.reshape(-1, ), residual2.reshape(-1, )

    # Function to compute the total loss (weighted sum of spatial boundary loss, temporal boundary loss and interior loss)
    def compute_loss(self, inp_train_sb_0, u_train_sb_0, inp_train_sb_1, u_train_sb_1, inp_train_tb, u_train_tb, inp_train_int, verbose=True):
        pred_Tf_0, pred_grad_Ts_x_0, pred_grad_Ts_x_1, pred_grad_Tf_x_1 = self.apply_boundary_conditions(inp_train_sb_0, inp_train_sb_1)
        pred_T_tb = self.apply_initial_condition(inp_train_tb)
        
        # compute the residuals
        r1_int, r2_int = self.compute_pde_residuals(inp_train_int)

        r_sb_1 = pred_grad_Ts_x_0
        r_sb_2 = pred_grad_Ts_x_1
        r_sb_3 = pred_grad_Tf_x_1
        r_sb_4 = pred_Tf_0 - u_train_sb_0[:,0] #Tf in first column of training data output 

        r_tb = u_train_tb - pred_T_tb

        loss_sb1 = torch.mean(abs(r_sb_1) ** 2)
        loss_sb2 = torch.mean(abs(r_sb_2) ** 2)
        loss_sb3 = torch.mean(abs(r_sb_3) ** 2)
        loss_sb4 = torch.mean(abs(r_sb_4) ** 2)
        loss_tb = torch.mean(abs(r_tb) ** 2)
        loss_int_1 = torch.mean(abs(r1_int) ** 2)
        loss_int_2 = torch.mean(abs(r2_int) ** 2)

        loss_boundary = loss_sb1 + loss_sb2 + loss_sb3 + loss_sb4 + loss_tb
        loss_interior = loss_int_1 + loss_int_2

        loss = torch.log10(self.lambda_u * loss_boundary + loss_interior)

        if verbose: print("Total loss: ", round(loss.item(), 4), "| PDE Loss: ", round(torch.log10(loss_boundary).item(), 4), "| Function Loss: ", round(torch.log10(loss_interior).item(), 4))

        return loss

    ################################################################################################
    def fit(self, num_epochs, optimizer, verbose=True):
        history = list()

        # Loop over epochs
        for epoch in range(num_epochs):
            if verbose: print("################################ ", epoch, " ################################")

            for j, ((inp_train_sb_0, u_train_sb_0), (inp_train_sb_1, u_train_sb_1), (inp_train_tb, u_train_tb), (inp_train_int, u_train_int)) in enumerate(zip(self.training_set_sb_0, self.training_set_sb_1 , self.training_set_tb, self.training_set_int)):
                def closure():
                    optimizer.zero_grad()
                    loss = self.compute_loss(inp_train_sb_0, u_train_sb_0, inp_train_sb_1, u_train_sb_1, inp_train_tb, u_train_tb, inp_train_int, verbose=verbose)
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

        output = self.approximate_solution(inputs)

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), dpi=150)
        im1 = axs[0].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output[:,0].detach(), cmap="jet")
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("t")
        plt.colorbar(im1, ax=axs[0])
        axs[0].grid(True, which="both", ls=":")

        im2 = axs[1].scatter(inputs[:, 0].detach(), inputs[:, 1].detach(), c=output[:,1].detach(), cmap="jet")
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("t")
        plt.colorbar(im2, ax=axs[1])
        axs[1].grid(True, which="both", ls=":")
        axs[0].set_title("Tf (fluid temperature)")
        axs[1].set_title("Ts (solid temperature)")

        plt.show()


# perform the training
n_int = 256
n_sb = 64
n_tb = 64

pinn = Pinns(n_int, n_sb, n_tb)

# Plot the input training points
input_sb_0_, output_sb_0_, input_sb_1_, output_sb_1_ = pinn.add_spatial_boundary_points()
input_tb_, output_tb_ = pinn.add_temporal_boundary_points()
input_int_, output_int_ = pinn.add_interior_points()

n_epochs = 5
optimizer_LBFGS = optim.LBFGS(pinn.approximate_solution.parameters(),
                              lr=float(0.5),
                              max_iter=50000,
                              max_eval=50000,
                              history_size=150,
                              line_search_fn="strong_wolfe",
                              tolerance_change=1.0 * np.finfo(float).eps)
optimizer_ADAM = optim.Adam(pinn.approximate_solution.parameters(),
                            lr=float(0.001))


hist = pinn.fit(num_epochs=n_epochs,
                optimizer=optimizer_LBFGS,
                verbose=True)

plt.figure(dpi=150)
plt.grid(True, which="both", ls=":")
plt.plot(np.arange(1, len(hist) + 1), hist, label="Train Loss")
plt.xscale("log")
plt.legend()
#plt.show()

pinn.plotting()

def evaluate_task_data(model):
    # write predictions for Tf and Ts into txt file
    tensor_data = np.loadtxt('TestingData.txt', delimiter=',', skiprows=1)
    tensor_data = torch.tensor(tensor_data, dtype=torch.float32)
    # data comes in format t, x
    with torch.no_grad():  # Disable gradient tracking since we're only predicting
        # switch order of t,x to x,t
        output = model(tensor_data[:, [1, 0]]) # only read out both Tf and Ts
    #print("max of Ts values: ", torch.max(output))
    # write predictions into file
    # Convert tensor data to list of lists
        # t x tf ts
    predictions = torch.cat((tensor_data, output), dim=1)
    list_data = predictions.tolist()
    # Define the file path
    file_path = "submission.txt"
    # Write data to text file
    with open(file_path, "w") as file:
        # Write the header
        file.write("t,x,tf,ts\n")
        # Write the data
        for row in list_data:
            file.write(",".join(map(str, row)) + "\n")
evaluate_task_data(pinn.approximate_solution)
print("file has been written")
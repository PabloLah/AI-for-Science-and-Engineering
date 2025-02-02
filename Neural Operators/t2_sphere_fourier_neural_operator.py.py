import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib.pyplot as plt
from neuralop import Trainer
from neuralop.datasets import load_spherical_swe
from neuralop.utils import count_model_params
from neuralop import LpLoss
from torch_harmonics import InverseRealSHT, RealSHT



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


################################################################
#  2d fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        nlat = x.shape[-2] 
        nlon = x.shape[-1] 
        sht = RealSHT(nlat, nlon, grid="equiangular")
        isht = InverseRealSHT(nlat, nlon, grid="equiangular")
        #apply spherical fourier
        x_ft = sht(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = isht(out_ft)
        return x

class SFNO2d(nn.Module):
    def __init__(self, sfno_architecture, device=None, padding_frac=1 / 4):
        super(SFNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution
        output shape: (batchsize, x=s, y=s, c=1)
        """
        self.modes1 = sfno_architecture["modes"]
        self.modes2 = sfno_architecture["modes"]
        self.width = sfno_architecture["width"]
        self.n_layers = sfno_architecture["n_layers"]
        self.retrain_sfno = sfno_architecture["retrain_sfno"]

        torch.manual_seed(self.retrain_sfno)
        self.padding_frac = padding_frac
        self.fc0 = nn.Linear(3, self.width)  # input channel is 3: (a(x, y), x, y)

        self.conv_list = nn.ModuleList(
            [nn.Conv2d(self.width, self.width, 1) for _ in range(self.n_layers)])
        self.spectral_list = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(self.n_layers)])

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

        self.to(device)

    def forward(self, x, **kwargs):
        x = x.permute(0, 2, 3, 1) #permute input from neurop library
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x1_padding = int(round(x.shape[-1] * self.padding_frac))
        x2_padding = int(round(x.shape[-2] * self.padding_frac))
        x = F.pad(x, [0, x1_padding, 0, x2_padding])

        for k, (s, c) in enumerate(zip(self.spectral_list, self.conv_list)):

            x1 = s(x)
            x2 = c(x)
            x = x1 + x2
            if k != self.n_layers - 1:
                x = F.gelu(x)
        x = x[..., :x.size(-2) - x2_padding, :x.size(-1) - x1_padding] #proposed modified cropping

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        x = x.permute(0, 3, 1, 2) #permute to match neurop library??

        return x

PERFORM_TRAINING = True
#Get Data
train_resolutions = [(32, 64), (64, 128), (128,256), (256,512), (512, 1024)]
test_resolutions = [(32, 64), (64, 128), (128,256), (256,512), (512, 1024)]

for train_resolution in train_resolutions:
    print(f'#############START TRAINING AT RESOLUTION {train_resolution} ###################')
    train_loader, test_loaders = load_spherical_swe(n_train=200, batch_size=4, train_resolution=train_resolution, test_resolutions=test_resolutions, n_tests=[50] * len(test_resolutions), test_batch_sizes=[10] * len(test_resolutions),)

    if PERFORM_TRAINING:
        epochs = 20
        #initialize model
        sfno_architecture = {
            "modes": 16,
            "width": 64,
            "n_layers": 4,
            "retrain_sfno": 0
        }
        sfno = SFNO2d(sfno_architecture)

        #----------TRAINING--------------
        n_params = count_model_params(sfno)
        print(f'\nOur model has {n_params} parameters.')

        l2loss = LpLoss(d=2, p=2, reduce_dims=(0,1))
        train_loss = l2loss
        eval_losses={'l2': l2loss} #'h1': h1loss,
        optimizer = torch.optim.Adam(sfno.parameters(),
                                        lr=8e-4,
                                        weight_decay=0.0)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        trainer = Trainer(model=sfno, n_epochs=epochs,
                        device=None,
                        wandb_log=False,
                        log_test_interval=3,
                        use_distributed=False,
                        verbose=True)
        trainer.train(train_loader=train_loader,
                    test_loaders={},
                    optimizer=optimizer,
                    scheduler=scheduler,
                    regularizer=False,
                    training_loss=train_loss,
                    eval_losses=eval_losses)

    # save and load model
    PATH = f"project_2/task_2/train_res_{train_resolution}/sfno_model.pt"
    SAVE_MODEL = True
    if SAVE_MODEL and PERFORM_TRAINING:
        torch.save(sfno, PATH)
        sfno = torch.load(PATH)
        print("model was saved.")

    LOAD_OLD_MODEL = False
    if LOAD_OLD_MODEL: 
        sfno = torch.load(PATH)


    #evaluate
    fig = plt.figure(figsize=(7, 7))
    for index, resolution in enumerate(test_resolutions):
        test_samples = test_loaders[resolution].dataset
        data = test_samples[0]
        x = data['x']
        y = data['y'][0, ...].numpy()
        x_in = x.unsqueeze(0)
        out = sfno(x_in)
        #adapt variables for plotting
        out = out.squeeze().detach().numpy()
        x = x[0, ...].detach().numpy()

        ax = fig.add_subplot(len(test_resolutions), 3, index*3 + 1)
        ax.imshow(x)
        ax.set_title(f'Input x {resolution}')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(len(test_resolutions), 3, index*3 + 2)
        ax.imshow(y)
        ax.set_title('Ground-truth y')
        plt.xticks([], [])
        plt.yticks([], [])

        ax = fig.add_subplot(len(test_resolutions), 3, index*3 + 3)
        ax.imshow(out)
        ax.set_title('Model prediction')
        plt.xticks([], [])
        plt.yticks([], [])
        
        #print mse error rate
        mse = np.mean((out.flatten() - y.flatten())**2)
        print(f'TRAIN_RES={train_resolution}. Testing at {resolution}, SFNO has an mse = ', mse)
    
    #safe performance plot for all resolutions
    fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
    plt.tight_layout()
    plt.savefig(f'project_2/task_2/train_res_{train_resolution}/plots/sfno_plots.png', dpi=300)
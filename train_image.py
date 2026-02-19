import os
import sys

import time
import numpy as np
from skimage import color
from tqdm import tqdm

import cv2
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torchvision.models as models
from torch.optim.lr_scheduler import LambdaLR

from modules.utils import normalize, measure, psnr

from fractions import Fraction

torch.cuda.set_device(7)


image = "kodim20"

input_img = f"data/Images/kodak/{image}.png"

save_dir = "./Results/images"
os.makedirs(save_dir, exist_ok=True)


# Model configs
niters = 500
learning_rate = 0.01  # Learning rate
decay_rate = 0.01 # Decay rate

act_params = 2
in_features = 2
out_features = 3
hidden_layers = 3
hidden_features = 256
maxpoints = 256*256

im = normalize(plt.imread(input_img).astype(np.float32), True)
H, W, _ = im.shape
if H > 2000 or W > 2000: scale = 1 / 2
else: scale = 1

im= cv2.resize(im, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

# im_gt_RGB = im

H, W, _ = im.shape
print(f"Image size: {H}x{W}")


x, y = torch.linspace(-1, 1, W), torch.linspace(-1, 1, H)
X, Y = torch.meshgrid(x, y, indexing='xy')
coords = torch.hstack((X.reshape(-1, 1), Y.reshape(-1, 1)))[None, ...]

gt = torch.tensor(im).cuda().reshape(H * W, 3)[None, ...]

img = gt
task = 'image'

mse_array = torch.zeros(niters).cuda()
time_array = torch.zeros_like(mse_array)
consts, psnr_vals, step_array = [], [], []

best_mse = torch.tensor(float('inf'))
best_img = None

rec = torch.zeros_like(gt)

# IMPORTANT: Set T_range and C_range constraints according to the task.
MLP_configs ={'task': task,
             'model': 'resnet34',
             'truncated_layer':5,
             'in_channels': 64,
             'hidden_channels': [64, 32, act_params*(hidden_layers+1)],
             'mlp_bias':0.3120,
             'activation_layer': nn.SiLU,
             'GT': torch.tensor(im).cuda()[None,...].permute(0, 3, 1, 2),
             'T_range': [5, 7],
             'c_range': [1, 2]
              } # Run for new range


class MLP(torch.nn.Sequential):
    '''
    Args:
        in_channels (int): Number of input channels or features.
        hidden_channels (list of int): List of hidden layer sizes. The last element is the output size.
        mlp_bias (float): Value for initializing bias terms in linear layers.
        activation_layer (torch.nn.Module, optional): Activation function applied between hidden layers. Default is SiLU.
        bias (bool, optional): If True, the linear layers include bias terms. Default is True.
        dropout (float, optional): Dropout probability applied after the last hidden layer. Default is 0.0 (no dropout).
    '''
    def __init__(self, MLP_configs, bias=True, dropout = 0.0):
        super().__init__()
        in_channels=MLP_configs['in_channels']
        hidden_channels=MLP_configs['hidden_channels']
        self.mlp_bias=MLP_configs['mlp_bias']
        activation_layer=MLP_configs['activation_layer']

        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(torch.nn.Linear(in_dim, hidden_dim, bias=bias))
            if MLP_configs['task'] == 'denoising':
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(activation_layer())
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(torch.nn.Dropout(dropout))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.001)
            torch.nn.init.constant_(m.bias, self.mlp_bias)

    def forward(self, x):
        self.consts = self.layers(x)
        return self.consts


class RaisedCosineImpulseResponseLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False,
                 beta0=0.5, eps=1e-8, out_real=False):
        super().__init__()

        self.beta0 = nn.Parameter(torch.tensor(beta0, dtype=torch.float), requires_grad=False)
        self.eps = eps
        self.is_first = is_first
        self.out_real = out_real

        self.in_features = in_features
        self.out_features = out_features

        dtype = torch.float if self.is_first else torch.cfloat
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)

        nn.init.uniform_(self.linear.weight, -1/self.in_features, 1/self.in_features)  # Initialize weights (new method)

    def forward(self, input, t0, c0):
        input = input.to(next(self.parameters()).device)  # Move input to correct device
        lin = self.linear(input)

        if not self.is_first:
            lin = lin / torch.abs(lin + self.eps)  # Normalize

        f1 = (1 / t0) * torch.sinc(lin / t0) * torch.cos(torch.pi * self.beta0 * lin / t0)
        f2 = 1 - (2 * self.beta0 * lin / t0) ** 2 + self.eps
        theta = 2 * torch.pi * c0 * lin * 1j

        rc = (f1 / f2)
        out = rc * torch.exp(theta)


        if not self.is_first:
            out = out / torch.abs(out + self.eps)

        return out.real if self.out_real else out

class INR(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features):
        super().__init__()

        self.img = MLP_configs['GT'].cuda()

        self.prior = MLP(MLP_configs)
        self.T_range = MLP_configs['T_range']
        self.c_range = MLP_configs['c_range']

        model_ft = getattr(models, MLP_configs['model'])(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(model_ft.children())[:MLP_configs['truncated_layer']])

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.nonlin = RaisedCosineImpulseResponseLayer

        dtype = torch.cfloat

        self.net = nn.ModuleList()  # Use ModuleList for proper CUDA movement
        self.net.append(self.nonlin(in_features, hidden_features, is_first=True))
        for _ in range(hidden_layers):
            self.net.append(self.nonlin(hidden_features, hidden_features))

        self.final_linear = nn.Linear(hidden_features, out_features, dtype=dtype)

    def forward(self, coords):
        extracted_features = self.feature_extractor(self.img)

        gap = self.gap(extracted_features.view(extracted_features.size(0), extracted_features.size(1), -1))
        coef = self.prior(gap[..., 0]).view(4,2)
        t0, c0 = torch.unbind(coef, dim=1)

        # ------------------------------- Sigmoid Projection ---------------------------
        t0 = torch.sigmoid(t0) * (self.T_range[1] - self.T_range[0]) + self.T_range[0]
        c0 = torch.sigmoid(c0) * (self.c_range[1] - self.c_range[0]) + self.c_range[0]
        # ------------------------------------------------------------------------------

        output = coords
        for id, lyr in enumerate(self.net):
            output = lyr(output, t0[id], c0[id])
        output = self.final_linear(output).real

        return nn.Sigmoid()(output)




model = INR(
    in_features=in_features,
    out_features=out_features,
    hidden_features=hidden_features,
    hidden_layers=hidden_layers)

model.cuda()

optim = torch.optim.Adam(lr=learning_rate * min(1, maxpoints / (H * W)),
                         params=model.parameters())

scheduler = LambdaLR(optim, lambda x: decay_rate ** min(x / niters, 1))
tot_iters = 0



print(f"Total Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")



def MSE_loss(x, y): return ((x - y) ** 2).mean()

init_time = time.time()
tbar = tqdm(range(niters))

# -------------------------------- Training Loop ---------------------------------
for epoch in (tbar):
    indices = torch.randperm(H * W)
    for b_idx in range(0, H*W, maxpoints):
        b_indices = indices[b_idx:min(H*W, b_idx+maxpoints)]
        b_coords = coords[:, b_indices, ...].cuda()
        b_indices = b_indices.cuda()

        pixelvalues = model(b_coords)
        with torch.no_grad(): rec[:, b_indices, :] = pixelvalues
        loss = MSE_loss(pixelvalues, img[:, b_indices, :])

        optim.zero_grad()
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

    time_array[epoch] = time.time() - init_time

    with torch.no_grad():
        mse_array[epoch] = ((gt - rec) ** 2).mean().item()

        im_gt = gt.reshape(H, W, 3).permute(2, 0, 1)[None, ...]
        im_rec = rec.reshape(H, W, 3).permute(2, 0, 1)[None, ...]

        consts.append(model.prior.consts.cpu().detach().numpy())

        psnrval = -10 * torch.log10(mse_array[epoch])
        psnr_vals.append(psnrval.cpu().numpy())

        tbar.set_description(f"Loss: {loss.item():.6f} "
                     f"| PSNR: {psnrval:.2f}")
        tbar.refresh()

    scheduler.step()
    imrec = rec[0, ...].reshape(H, W, 3).detach().cpu().numpy()

    if (mse_array[epoch] < best_mse) or (epoch == 0):
        best_mse = mse_array[epoch]
        best_img = imrec
        best_epoch = epoch
        best_model_state = model.state_dict()

    tot_iters+=1



best_img = best_img.astype(np.float32)
im = im.astype(np.float32) # Ensure float32 data type

work_dir = f"{save_dir}/{image}"
os.makedirs(work_dir, exist_ok=True)

weight_path = os.path.join(work_dir, f"{image}_iters_{tot_iters}_psnr_{max(psnr_vals):.2f}.pth")
torch.save({'state_dict': best_model_state,
            'best_epoch': best_epoch,
            'gt': im,
            'rec': best_img,
            'consts_array': np.array(consts),
            'time_array': time_array.detach().cpu().numpy(),
            'mse_array': mse_array.detach().cpu().numpy(),
            'psnr_vals': np.array(psnr_vals)
            }, weight_path)


fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the first image
axes[0].imshow(im)
#axes[0].imshow(im)
axes[0].set_title('Ground Truth')
axes[0].axis('off')  # Hide axis

# Display the second image
axes[1].imshow(best_img)
#axes[1].imshow(best_img)
axes[1].set_title(f"Reconstructed Image: PSNR {max(psnr_vals):.4f}")#n RGB PSNR {(rgb_psnr):.4f}")
axes[1].axis('off')  # Hide axis

# Save figs
plt.savefig(f"{work_dir}/{image}_iters_{tot_iters}_psnr_{max(psnr_vals):.2f}_comparison.png")

plt.close()


def get_sigmoid_projected_params(A, T_range=[0,10], c_range=[0,3]):
    """
    Given unconstrained parameters A, project them via sigmoid to desired ranges.
    Args:
        A: tensor of shape (Tunable_params,)
        T_range: list, desired range for time parameter t0
        c_range: list, desired range for concentration parameter c0
    Returns:
        t0: projected time parameter in T range [0, 10]
        c0: projected concentration parameter in C range [0, 3]
    """
    A_ = A.view(4,2)
    t0, c0 = torch.unbind(A_, dim=1)
    t0 = torch.sigmoid(t0) * (T_range[1] - T_range[0]) + T_range[0]
    c0 = torch.sigmoid(c0) * (c_range[1] - c_range[0]) + c_range[0]
    return t0, c0


final_coverged_A = np.array(consts)[-1,:,:]
t0, c0 = get_sigmoid_projected_params(torch.tensor(final_coverged_A), T_range=MLP_configs['T_range'], c_range=MLP_configs['c_range'])

print("Final projected parameters after training:")
print("t0:", t0)
print("c0:", c0)




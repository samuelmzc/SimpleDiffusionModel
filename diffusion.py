import numpy as np
import matplotlib.pyplot as plt
from sympy import diff
from sympy.tensor import get_indices
import torch as torch
from process_data import tensor_to_pil


unsqueeze3x = lambda x : x[..., None, None, None] 

class diffusion_cosine:
    def __init__(self, timesteps):
        self.timesteps = timesteps
        self.alpha_bar_schedule = (
            lambda t: np.cos((t/timesteps + 0.008)/(1 + 0.008) * np.pi/2)**2
        )
        diff_params = self.params(self.alpha_bar_schedule)
        self.beta, self.alpha, self.alpha_bar = diff_params["betas"], diff_params["alphas"], diff_params["alphas_bar"]
        self.beta_tilde = self.beta[1:] * (
            (1 - self.alpha_bar[:-1])
            /
            (1 - self.alpha_bar[1:])
        )
        self.beta_tilde = torch.cat(
            [self.beta_tilde[0:1], self.beta_tilde]
        )

    def params(self, scheduler):
        diff_params = {}
        diff_params["betas"] = torch.from_numpy(
            np.array(
                [
                    min(
                        1 - scheduler(t + 1)/scheduler(t),
                        0.999,
                    )
                    for t in range(self.timesteps)
                ]
            )
        )
        diff_params["alphas"] = 1 - diff_params["betas"]
        diff_params["alphas_bar"] = torch.cumprod(diff_params["alphas"], dim = 0)
        return diff_params

    def forward(self, x0, t):
        noise = torch.randn_like(x0)
        xt = (
            unsqueeze3x(torch.sqrt(self.alpha_bar[t])) * x0
            +
            unsqueeze3x(torch.sqrt(1 - self.alpha_bar[t])) * noise
        )
        return xt.float(), noise

    def sample(self, xT, model, timesteps = None, save = None):
        model.eval()

        timesteps = timesteps or self.timesteps
        sub_timesteps = np.linspace(0, timesteps - 1, timesteps)
        xt = xT
        
        for i, t in zip(np.arange(timesteps)[::-1], sub_timesteps[::-1]):
            with torch.no_grad():
                current_t = torch.full((1,), t)
                current_t_indexed = torch.full((1,), i)
                noise_pred = model(xt, current_t)

                mean = (
                    1
                    /
                    unsqueeze3x(self.alpha[current_t_indexed].sqrt())
                    *
                    (xt - (
                        unsqueeze3x(self.beta[current_t_indexed])
                        /
                        unsqueeze3x((1 - self.alpha_bar[current_t_indexed]).sqrt())
                    ) * noise_pred)
                )
        
                if i == 0:
                    xt = mean
                
                else:
                    xt = mean + unsqueeze3x(self.beta_tilde[current_t_indexed].sqrt()) * torch.randn_like(xt)
                
                xt = xt.float()

                if save:
                    img = tensor_to_pil(xt)
                    plt.imshow(img, cmap = "gray")
                    plt.axis(False)
                    plt.title(f"x_{i + 1}")
                    plt.savefig(f"results/reversed_{i + 1 - 100}.png")

        return xt.detach().float()

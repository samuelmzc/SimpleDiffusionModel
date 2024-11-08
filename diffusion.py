import torch as torch

def get_index_from_list(vals, t, x_shape):
    """
    Returns specific index t of a passed list of values vals, and considers the batch size
    El tensor out se reconvierte en uno con forma (batch_size, 1, 1, .... , 1), dim = x_shape
    La operación ((1,) * (len(x_shape) - 1)) genera una tupla de números 1 del tamaño len(x_shape)-1
    y * desempaqueta la tupla para convertirla en valores individuales
    """

    batch_size = t.shape[0]
    out = vals.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class diffusion:
    def __init__(self, timesteps, start = 10**(-7), end = 0.02):
        self.beta = torch.linspace(start, end, timesteps).sin()
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, axis = 0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        
    def forward_step(self, x, t):
        noise = torch.randn_like(x)
        sqrt_alpha_bar_t = get_index_from_list(self.sqrt_alpha_bar, t, x.shape)
        sqrt_alpha_bar_minus1 = get_index_from_list(torch.sqrt(1 - self.alpha_bar), t, x.shape)
        xt = sqrt_alpha_bar_t * x + sqrt_alpha_bar_minus1 * noise
        return xt, noise

    def backward_step(self, x, t, model):
        beta_t = get_index_from_list(self.beta, t, x.shape)
        alpha_bar_t = get_index_from_list(self.alpha_bar, t, x.shape)
        sqrt_alpha_recip_t = get_index_from_list(1 / torch.sqrt(self.alpha), t, x.shape)
        sqrt_alpha_minus1_t = get_index_from_list(torch.sqrt(1 - self.alpha_bar), t, x.shape)
        

        model_mean = sqrt_alpha_recip_t * (x - beta_t/sqrt_alpha_minus1_t * model.forward(x, t))
        
        if t == 0:
            return model_mean
        else:
            alpha_bar_tminus1 = get_index_from_list(self.alpha_bar, t - 1, x.shape)
            z = torch.randn_like(x)
            posterior_variance = (1 - alpha_bar_tminus1)/(1 - alpha_bar_t) * beta_t
            return model_mean + torch.sqrt(posterior_variance)*z


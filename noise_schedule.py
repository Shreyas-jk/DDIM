import torch


class NoiseSchedule:
    def __init__(self, T=1000, beta_start=0.0001, beta_end=0.02):

        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)

        self.alphas = 1 - self.betas

        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bars)

        self.sqrt_recip_alpha = torch.sqrt(1 / self.alphas)
        self.beta_over_sqrt_one_minus_alpha_bar = self.betas / self.sqrt_one_minus_alpha_bar

    def add_noise(self, x_0, t, noise):
        # Equation 4 - jump to any noise level instantly
        # x_t = sqrt(alpha_bar_t) * clean_image + sqrt(1 - alpha_bar_t) * noise

        # Reshape constants from (batch_size,) to (batch_size, 1, 1, 1)
        # so they broadcast against images of shape (batch_size, channels, height, width)
        a = self.sqrt_alpha_bar[t].reshape(-1, 1, 1, 1)
        b = self.sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1)

        return a * x_0 + b * noise

    def denoise_one_step(self, x_t, noise_pred, t):
        # Algorithm 2 - reverse one diffusion step
        # x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (beta_t/sqrt(1-alpha_bar_t)) * predicted_noise) + sigma_t * z

        a = self.sqrt_recip_alpha[t].reshape(-1, 1, 1, 1)
        b = self.beta_over_sqrt_one_minus_alpha_bar[t].reshape(-1, 1, 1, 1)

        # The deterministic part - remove predicted noise
        mean = a * (x_t - b * noise_pred)

        # Add randomness (except at the very last step t=0)
        if t > 0:
            sigma = torch.sqrt(self.betas[t]).reshape(-1, 1, 1, 1)
            random_noise = torch.randn_like(x_t)
            return mean + sigma * random_noise
        else:
            return mean

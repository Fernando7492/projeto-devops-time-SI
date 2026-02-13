import torch
import numpy as np

class CommunicationSystem:
    def __init__(self, msg_len, device):
        self.msg_len = msg_len
        self.device = device

    def generate_bits(self, batch_size):
        return torch.randint(0, 2, (batch_size, self.msg_len)).float().to(self.device)

    def modulate_bpsk(self, bits):
        return 2 * bits - 1

    def apply_v2x_channel(self, tx_signal, snr_db):
        batch, length = tx_signal.shape
        h_real = torch.randn(batch, length).to(self.device)
        h_imag = torch.randn(batch, length).to(self.device)
        factor = torch.sqrt(h_real**2 + h_imag**2)
        h_real = h_real / factor
        h_imag = h_imag / factor
        rx_real = tx_signal * h_real
        rx_imag = tx_signal * h_imag
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        noise_r = torch.randn_like(rx_real) * noise_std
        noise_i = torch.randn_like(rx_imag) * noise_std
        y_real = rx_real + noise_r
        y_imag = rx_imag + noise_i
        return y_real, y_imag, h_real, h_imag
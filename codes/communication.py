import torch


class CommunicationSystem:
    def __init__(self, msg_len, device):
        self.msg_len = msg_len
        self.device = device

    def generate_bits(self, batch_size):
        return torch.randint(0, 2, (batch_size, self.msg_len)).float().to(self.device)

    def modulate_bpsk(self, bits):
        return 2 * bits - 1

    def apply_awgn(self, signal, snr_db):
        snr_linear = 10 ** (snr_db / 10.0)
        power_signal = torch.mean(signal ** 2)
        noise_power = power_signal / snr_linear
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(signal) * noise_std
        return signal + noise

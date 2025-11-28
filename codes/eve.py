import torch


def avaliar_eve(model, comm, bits, snr_eve):
    tx = comm.modulate_bpsk(bits)
    rx = comm.apply_awgn(tx, snr_eve)
    pred = (model(rx) > 0.5).float()

    erros = torch.sum(torch.abs(pred - bits))
    ber = erros / bits.numel()

    return float(ber)

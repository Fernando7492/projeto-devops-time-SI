import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import os

from models import PLS_DNN
from communication import CommunicationSystem
from fec_utils import encode_repetition, decode_repetition

MESSAGE_LEN = 2100 
BATCH_SIZE = 128        
NUM_EPOCHS = 1000
LEARNING_RATE = 0.005
SNR_LIST = [-15, -10, -5, 0, 5, 10, 15, 20] 
FEC_N = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_simulation():
    os.makedirs('results', exist_ok=True)
    
    comm_sys = CommunicationSystem(MESSAGE_LEN, DEVICE)
    bob_model = PLS_DNN().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(bob_model.parameters(), lr=LEARNING_RATE)
    
    print(f"Hardware: {DEVICE}")
    print("\n--- TREINAMENTO ---")
    bob_model.train()
    for epoch in range(NUM_EPOCHS):
        bits = comm_sys.generate_bits(BATCH_SIZE)
        tx = comm_sys.modulate_bpsk(bits)
        rx_r, rx_i, h_r, h_i = comm_sys.apply_v2x_channel(tx, snr_db=10)
        inputs = torch.stack((rx_r, rx_i, h_r, h_i), dim=2).view(-1, 4)
        targets = bits.view(-1, 1)
        optimizer.zero_grad()
        loss = criterion(bob_model(inputs), targets)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Loss: {loss.item():.6f}")

    print("\n--- SIMULAÇÃO COMPARATIVA ---")
    print(f"{'SNR':>5} | {'Bob Raw':>10} | {'Bob FEC':>10} | {'Eve Raw':>10} | {'Eve FEC':>10}")
    print("-" * 65)

    ber_bob_raw, ber_eve_raw = [], []
    ber_bob_fec, ber_eve_fec = [], []
    
    bob_model.eval()
    for snr in SNR_LIST:
        e_b_r, e_e_r, e_b_f, e_e_f, t_r, t_f = 0, 0, 0, 0, 0, 0
        for _ in range(50):
            with torch.no_grad():
                msg_len_raw = MESSAGE_LEN // FEC_N
                bits_orig = torch.randint(0, 2, (100, msg_len_raw)).float().to(DEVICE)
                bits_coded = encode_repetition(bits_orig, n=FEC_N)
                tx = comm_sys.modulate_bpsk(bits_coded)
                rx_r, rx_i, h_r, h_i = comm_sys.apply_v2x_channel(tx, snr)
                inputs = torch.stack((rx_r, rx_i, h_r, h_i), dim=2).view(-1, 4)
                preds = bob_model(inputs).view_as(bits_coded)
                d_bob = (preds > 0.5).float()
                d_eve = (rx_r > 0).float()
                e_b_r += torch.sum(torch.abs(d_bob - bits_coded)).item()
                e_e_r += torch.sum(torch.abs(d_eve - bits_coded)).item()
                t_r += bits_coded.numel()
                f_bob = decode_repetition(d_bob, n=FEC_N)
                f_eve = decode_repetition(d_eve, n=FEC_N)
                e_b_f += torch.sum(torch.abs(f_bob - bits_orig)).item()
                e_e_f += torch.sum(torch.abs(f_eve - bits_orig)).item()
                t_f += bits_orig.numel()

        b_r, e_r, b_f, e_f = e_b_r/t_r, e_e_r/t_r, e_b_f/t_f, e_e_f/t_f
        ber_bob_raw.append(b_r); ber_eve_raw.append(e_r)
        ber_bob_fec.append(b_f); ber_eve_fec.append(e_f)
        print(f"{snr:3d}dB | {b_r:10.5f} | {b_f:10.5f} | {e_r:10.5f} | {e_f:10.5f}")

    plt.figure(figsize=(10, 7))
    plt.semilogy(SNR_LIST, ber_bob_raw, 'b--', label='Bob (IA)')
    plt.semilogy(SNR_LIST, ber_bob_fec, 'b-o', linewidth=2, label='Bob (IA + FEC)')
    plt.semilogy(SNR_LIST, ber_eve_raw, 'r--', label='Eve (IA)')
    plt.semilogy(SNR_LIST, ber_eve_fec, 'r-s', linewidth=2, label='Eve (IA + FEC)')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.xlabel("SNR (dB)"); plt.ylabel("BER")
    plt.legend(); plt.savefig("results/comparative_result.png")
    print("\nSUCESSO: Gráfico salvo em 'results/comparative_result.png'")

if __name__ == "__main__":
    run_simulation()
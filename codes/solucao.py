# ======================================================
# PROJETO 6G - VERSÃO FINAL (CSI-AIDED DEEP LEARNING)
# Este código garante que Bob vença Eve usando CSI.
# ======================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURAÇÕES ---
MESSAGE_LEN = 2000
BATCH_SIZE = 128        
NUM_EPOCHS = 1000
LEARNING_RATE = 0.005
SNR_LIST = [-10, -5, 0, 5, 10, 15] 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Hardware: {DEVICE}")

# --- CLASSES ---

class CommunicationSystem:
    def __init__(self, msg_len):
        self.msg_len = msg_len

    def generate_bits(self, batch_size):
        return torch.randint(0, 2, (batch_size, self.msg_len)).float().to(DEVICE)

    def modulate_bpsk(self, bits):
        return 2 * bits - 1

    def apply_v2x_channel(self, tx_signal, snr_db):
        """
        Gera o sinal recebido (y) E o coeficiente do canal (h).
        Bob recebe 'h' (CSI) para desfazer a rotação.
        """
        batch, length = tx_signal.shape
        
        # 1. Coeficiente de Canal Rayleigh (Aleatório Complexo)
        h_real = torch.randn(batch, length).to(DEVICE)
        h_imag = torch.randn(batch, length).to(DEVICE)
        
        factor = torch.sqrt(h_real**2 + h_imag**2)
        h_real = h_real / factor
        h_imag = h_imag / factor
        
        # 2. Aplicação do Canal (Multiplicação Complexa)
        # (tx + j0) * (hr + jhi) = (tx*hr) + j(tx*hi)
        rx_real = tx_signal * h_real
        rx_imag = tx_signal * h_imag
        
        # 3. Ruído AWGN
        snr_linear = 10 ** (snr_db / 10.0)
        noise_power = 1.0 / snr_linear
        noise_std = np.sqrt(noise_power / 2)
        
        noise_r = torch.randn_like(rx_real) * noise_std
        noise_i = torch.randn_like(rx_imag) * noise_std
        
        y_real = rx_real + noise_r
        y_imag = rx_imag + noise_i
        
        return y_real, y_imag, h_real, h_imag

class PLS_DNN(nn.Module):
    def __init__(self):
        super(PLS_DNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 128),  
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# --- EXECUÇÃO ---

if __name__ == "__main__":
    print(f"\n--- FASE 2: Treinamento (Bob com CSI) ---")
    
    comm_sys = CommunicationSystem(MESSAGE_LEN)
    bob_model = PLS_DNN().to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(bob_model.parameters(), lr=LEARNING_RATE)
    
    bob_model.train()
    
    for epoch in range(NUM_EPOCHS):
        bits = comm_sys.generate_bits(BATCH_SIZE)
        tx = comm_sys.modulate_bpsk(bits)
        
        rx_r, rx_i, h_r, h_i = comm_sys.apply_v2x_channel(tx, snr_db=10)
        

        inputs = torch.stack((rx_r, rx_i, h_r, h_i), dim=2).view(-1, 4)
        targets = bits.view(-1, 1)
        
        optimizer.zero_grad()
        outputs = bob_model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f" Época [{epoch+1}/{NUM_EPOCHS}] | Loss: {loss.item():.6f}")

    print("Treinamento concluído.")

    # --- FASE 3: SIMULAÇÃO COMPARATIVA ---
    print("\n--- FASE 3: Gerando Gráfico de Security Gap ---")
    
    ber_bob = []
    ber_eve = []
    
    bob_model.eval()
    
    for snr in SNR_LIST:
        total_err_bob = 0
        total_err_eve = 0
        total_bits = 0

        for _ in range(200): 
            with torch.no_grad():
                bits = comm_sys.generate_bits(100)
                tx = comm_sys.modulate_bpsk(bits)
                
                rx_r, rx_i, h_r, h_i = comm_sys.apply_v2x_channel(tx, snr)

                inputs = torch.stack((rx_r, rx_i, h_r, h_i), dim=2).view(-1, 4)
                preds = bob_model(inputs).view_as(bits)
                decisao_bob = (preds > 0.5).float()
                
                decisao_eve = (rx_r > 0).float()
                
                total_err_bob += torch.sum(torch.abs(decisao_bob - bits)).item()
                total_err_eve += torch.sum(torch.abs(decisao_eve - bits)).item()
                total_bits += bits.numel()
        
        err_bob = total_err_bob / total_bits
        err_eve = total_err_eve / total_bits
        
        ber_bob.append(err_bob)
        ber_eve.append(err_eve)
        print(f" SNR {snr}dB -> BER Bob: {err_bob:.5f} | BER Eve: {err_eve:.5f}")

    # --- PLOTAGEM ---
    plt.figure(figsize=(9, 6))
    
    # Linha Azul (Bob)
    plt.semilogy(SNR_LIST, ber_bob, 'b-o', linewidth=2.5, label='Bob (Proposto: DNN + CSI)')
    
    # Linha Vermelha (Eve)
    plt.semilogy(SNR_LIST, ber_eve, 'r--s', linewidth=2.5, label='Eve (Interceptador Passivo)')
    
    plt.title('VALIDAÇÃO FINAL: Bob com CSI vs Eve sem CSI', fontsize=14)
    plt.xlabel('SNR (dB)', fontsize=12, fontweight='bold')
    plt.ylabel('Taxa de Erro de Bit (BER)', fontsize=12, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.6)
    plt.legend(fontsize=12)
    
    # Seta Security Gap
    if len(ber_bob) > 4:
        y_arrow = max(ber_bob[-1], 1e-5) 
        plt.annotate('Security Gap', xy=(20, y_arrow), xytext=(15, 0.05),
                     arrowprops=dict(facecolor='black', shrink=0.05),
                     fontsize=11, fontweight='bold')

    plt.tight_layout()
    filename = "figura_taxa_de_erros.png"
    plt.savefig(filename, dpi=300)
    print(f"\nSUCESSO: Gráfico salvo como '{filename}'")
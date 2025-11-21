# ======================================================
# PROJETO DE SEGURANÇA EM REDES 6G - ENTREGA 3 (POC)
# Framework: PLS-DNN (Physical Layer Security with Deep Learning)
# Autores: Fernando, Emanuel, Pedro W., Gustavo, Pedro J.
# ======================================================

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- 1. CONFIGURAÇÕES GLOBAIS (MODO CIENTÍFICO) ---
# Hiperparâmetros ajustados para convergência real
MESSAGE_LEN = 1000      # Bits por pacote
BATCH_SIZE = 64         # Pacotes por lote
NUM_EPOCHS = 500        # AUMENTADO: Mais tempo para a IA aprender o padrão
LEARNING_RATE = 0.002   # AJUSTADO: Leve aumento para acelerar convergência inicial
SNR_TRAIN_DB = 10       # MANTIDO: Cenário desafiador (não facilitamos o ruído)

# Seleção de Hardware
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f" Hardware: GPU NVIDIA detectada: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print(" Hardware: Rodando em CPU.")

# --- 2. CLASSES DO SISTEMA ---

class CommunicationSystem:
    def __init__(self, msg_len):
        self.msg_len = msg_len

    def generate_bits(self, batch_size):
        """Alice: Gera bits aleatórios"""
        return torch.randint(0, 2, (batch_size, self.msg_len)).float().to(DEVICE)

    def modulate_bpsk(self, bits):
        """Modulação BPSK"""
        return 2 * bits - 1

    def apply_awgn(self, signal, snr_db):
        """Canal com Ruído Branco Gaussiano"""
        snr_linear = 10 ** (snr_db / 10.0)
        power_signal = torch.mean(signal ** 2)
        noise_power = power_signal / snr_linear
        noise_std = torch.sqrt(noise_power)
        noise = torch.randn_like(signal) * noise_std
        return signal + noise

class PLS_DNN(nn.Module):
    def __init__(self, input_size):
        super(PLS_DNN, self).__init__()
        # Arquitetura
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# --- 3. EXECUÇÃO PRINCIPAL ---

if __name__ == "__main__":
    print(f"\n--- Iniciando Simulação (Modo Científico: {NUM_EPOCHS} Épocas) ---")

    comm_sys = CommunicationSystem(MESSAGE_LEN)
    bob_model = PLS_DNN(MESSAGE_LEN).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(bob_model.parameters(), lr=LEARNING_RATE)

    loss_history = []

    bob_model.train()

    # Loop de Treinamento
    for epoch in range(NUM_EPOCHS):
        bits_originais = comm_sys.generate_bits(BATCH_SIZE)
        sinal_tx = comm_sys.modulate_bpsk(bits_originais)
        sinal_rx = comm_sys.apply_awgn(sinal_tx, SNR_TRAIN_DB)

        bits_preditos_prob = bob_model(sinal_rx)

        loss = criterion(bits_preditos_prob, bits_originais)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())

        # Log menos frequente para não poluir o terminal
        if (epoch + 1) % 50 == 0:
            print(f"  Época [{epoch+1}/{NUM_EPOCHS}] | Perda (Loss): {loss.item():.6f}")

    print("Treinamento concluído.")

    # --- 4. VALIDAÇÃO ---
    print("\n--- Gerando Métricas Finais ---")
    bob_model.eval()
    with torch.no_grad():
        test_bits = comm_sys.generate_bits(100)
        test_tx = comm_sys.modulate_bpsk(test_bits)
        test_rx = comm_sys.apply_awgn(test_tx, SNR_TRAIN_DB)

        pred_prob = bob_model(test_rx)
        pred_bits = (pred_prob > 0.5).float()

        bit_errors = torch.sum(torch.abs(pred_bits - test_bits))
        total_bits = test_bits.numel()
        ber = bit_errors / total_bits

    print(f"\nRESULTADOS FINAIS (SNR = {SNR_TRAIN_DB} dB):")
    print(f"  Total Bits: {total_bits}")
    print(f"  Bits Errados: {int(bit_errors)}")
    print(f"  BER (Taxa de Erro): {ber:.6f}")

    # --- 5. SALVAR GRÁFICO (SEM INTERFACE GRÁFICA) ---
    print("\nGerando gráfico...")
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Erro de Treinamento', color='blue')
    plt.title(f'Convergência do PLS-DNN (SNR={SNR_TRAIN_DB}dB - 500 Épocas)')
    plt.xlabel('Épocas')
    plt.ylabel('Loss (Entropia Binária)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Salva o arquivo em vez de tentar abrir a janela
    nome_arquivo = "resultado_treinamento_v2.png"
    plt.savefig(nome_arquivo)
    print(f"SUCESSO: Gráfico salvo como '{nome_arquivo}' na pasta atual.")

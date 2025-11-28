# ======================================================
# PROJETO DE SEGURANÇA EM REDES 6G - ENTREGA 6
# Framework: PLS-DNN (Physical Layer Security with Deep Learning)
# Autores: Fernando, Emanuel, Pedro W., Gustavo, Pedro J.
# ======================================================

import torch
import matplotlib.pyplot as plt
import csv

from communication import CommunicationSystem
from bob import BobPLSDNN, treinar_bob
from eve import avaliar_eve

# --------------------
# CONFIGS
# --------------------
MESSAGE_LEN = 256
BATCH_SIZE = 128
NUM_EPOCHS = 600
LEARNING_RATE = 0.0015

SNR_BOB_DB = 10
SNR_EVE_DB = 0


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Hardware ativo:", DEVICE)

# --------------------
# INSTÂNCIAS
# --------------------
comm = CommunicationSystem(MESSAGE_LEN, DEVICE)
bob = BobPLSDNN(MESSAGE_LEN).to(DEVICE)

# --------------------
# TREINAMENTO
# --------------------
print("\n--- Iniciando Treinamento ---\n")
losses = treinar_bob(
    model=bob,
    comm=comm,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    snr=SNR_BOB_DB,
    lr=LEARNING_RATE
)

# --------------------
# AVALIAÇÃO (BOB & EVE)
# --------------------
print("\n--- Avaliação com Bob e Eve ---")
bob.eval()

with torch.no_grad():
    bits_teste = comm.generate_bits(1000)

    # Bob
    tx = comm.modulate_bpsk(bits_teste)
    rx_bob = comm.apply_awgn(tx, SNR_BOB_DB)
    pred_bob = (bob(rx_bob) > 0.5).float()
    ber_bob = float(
        torch.sum(torch.abs(pred_bob - bits_teste)) / bits_teste.numel())

    # Eve
    ber_eve = avaliar_eve(bob, comm, bits_teste, SNR_EVE_DB)

print(f"SNR Bob: {SNR_BOB_DB} dB -> BER Bob: {ber_bob:.6f}")
print(f"SNR Eve: {SNR_EVE_DB} dB -> BER Eve: {ber_eve:.6f}")

security_gap = SNR_BOB_DB - SNR_EVE_DB
print(f"Security Gap: {security_gap} dB")

# --------------------
# SALVAR CSV
# --------------------
print("\nSalvando métricas em resultados_snr.csv ...")

with open("resultados_snr.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["snr_bob", "ber_bob", "snr_eve",
                    "ber_eve", "security_gap"])
    writer.writerow([SNR_BOB_DB, ber_bob, SNR_EVE_DB, ber_eve, security_gap])

print("Arquivo resultados_snr.csv gerado com sucesso.")

# --------------------
# GRÁFICO
# --------------------
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Convergência do modelo")
plt.xlabel("Épocas")
plt.ylabel("Loss")
plt.grid(True)
plt.savefig("resultado_treinamento.png")

print("Gráfico salvo como resultado_treinamento.png")

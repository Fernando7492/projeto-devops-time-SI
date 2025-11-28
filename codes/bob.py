import torch
import torch.nn as nn
import torch.optim as optim


class BobPLSDNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, input_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def treinar_bob(model, comm, epochs, batch_size, snr, lr):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    losses = []

    for epoch in range(epochs):
        bits = comm.generate_bits(batch_size)
        tx = comm.modulate_bpsk(bits)
        rx = comm.apply_awgn(tx, snr)

        pred = model(rx)
        loss = criterion(pred, bits)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (epoch + 1) % 50 == 0:
            print(f"Época [{epoch+1}/{epochs}] | Loss: {loss.item():.6f}")

    return losses

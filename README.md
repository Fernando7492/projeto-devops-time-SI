# PLS-DNN: Segurança de Camada Física em Redes 6G com Deep Learning

![Status](https://img.shields.io/badge/Status-Em%20Desenvolvimento-yellow)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Framework](https://img.shields.io/badge/Framework-PyTorch-EE4C2C)

Repositório oficial do projeto da disciplina de **Segurança da Informação** (2025.2) da **Universidade Federal do Agreste de Pernambuco (UFAPE)**.

## 📄 Sobre o Projeto

Este projeto investiga a aplicação de **Deep Learning** para garantir a segurança na camada física (Physical Layer Security - PLS) em redes 6G.

O objetivo é validar a eficácia de uma **Rede Neural Profunda (DNN)** na decodificação de sinais em comparação com métodos tradicionais, focando especialmente na lacuna de pesquisa referente a cenários de alta mobilidade e canais não-estacionários.

A implementação atual consiste em uma **Prova de Conceito (PoC)** que simula:
1.  **Transmissor (Alice):** Geração de bits e modulação BPSK.
2.  **Canal:** Simulação de ruído AWGN (Additive White Gaussian Noise) baseada no modelo COST 259.
3.  **Receptor Inteligente (Bob):** Uma DNN treinada para decodificar o sinal ruidoso e corrigir erros de bit (BER).

## 📂 Estrutura do Repositório

```bash
.
├── codes/                      # Código-fonte da simulação
│   ├── solucao.py              # Script principal (Treinamento e Validação)
│   └── resultado_treinamento_v2.png # Gráfico de convergência gerado
├── docs/                       # Documentação acadêmica e artefatos
│   ├── Arquitetura_Simulacao.png
│   ├── Relatórios e Apresentações...
├── files/                      # Arquivos auxiliares (LaTeX, Referências)
└── README.md
```

## 🚀 Como Rodar Localmente

Siga os passos abaixo para clonar e executar a simulação no seu ambiente Linux.

### 1. Pré-requisitos
Certifique-se de ter o **Python 3** e o **Git** instalados.

### 2. Clonar o Repositório
```bash
git clone [https://github.com/fernando7492/projeto-devops-time-SI.git](https://github.com/fernando7492/projeto-devops-time-SI.git)
cd projeto-devops-time-SI
```

### 3. Configurar o Ambiente Virtual
Recomendamos usar um ambiente virtual (`venv`) para isolar as dependências.

```bash
# Criar o ambiente virtual (na pasta oculta .venv)
python3 -m venv .venv

# Ativar o ambiente
source .venv/bin/activate
```

### 4. Instalar Dependências
Instale as bibliotecas necessárias (`torch`, `numpy`, `scipy`, `matplotlib`).

> **Nota:** Se você não possui uma GPU dedicada ou tem pouco espaço em disco, use o comando abaixo para instalar a versão leve (CPU-only) do PyTorch:

```bash
pip install torch --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
pip install numpy scipy matplotlib
```

### 5. Executar a Simulação
Navegue até a pasta de códigos e execute o script:

```bash
cd codes
python solucao.py
```

### 🔍 O que esperar da execução?
1.  O script detectará automaticamente seu hardware (CPU ou GPU).
2.  Iniciará o treinamento da rede neural por **500 épocas**.
3.  Exibirá a redução da função de perda (*Loss*) no terminal.
4.  Ao final, calculará a **Taxa de Erro de Bit (BER)**.
5.  Salvará um gráfico de convergência como `resultado_treinamento_v2.png` na pasta atual.

---

## 📊 Resultados Preliminares

Abaixo, um exemplo da curva de aprendizado do receptor (Bob), demonstrando a capacidade da rede de reduzir a entropia e aprender a corrigir os erros do canal ruidoso.

![Gráfico de Convergência](codes/resultado_treinamento_v2.png)

---

## 👥 Equipe

* **Emanuel Reino**
* **Fernando Emidio**
* **Gustavo Wanderley**
* **Pedro William**
* **Pedro José**

---
Desenvolvido no contexto acadêmico da UFAPE.

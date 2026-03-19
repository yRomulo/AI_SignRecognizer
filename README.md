

# 🚀 AI SignRecognizer

[![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![FastHTML](https://img.shields.io/badge/FastHTML-Web%20App-000000?style=for-the-badge&logo=fastapi)](#)

O **AI SignRecognizer** é uma aplicação completa de visão computacional projetada para rastrear e reconhecer gestos manuais em tempo real através da webcam. O sistema une **OpenCV** e **MediaPipeLine** para o rastreamento das mãos, utiliza algoritmos clássicos de Machine Learning via **Scikit-Learn** para classificação, e entrega os resultados dinamicamente usando **FastHTML** sobre **WebSockets**.

---

## ✨ Funcionalidades Principais

- 🖐️ **Detecção Multi-Mão Instantânea**: Rastreamento simultâneo dos 21 marcos anatômicos para 1 ou 2 mãos na tela.
- ⚡ **Performance via WebSockets**: Streaming contínuo de vídeo do frontend (no navegador) para o backend responsável pela predição, com latência minimizada.
- 🧠 **IA Customizável pelo Usuário**: Os desenvolvedores podem criar facilmente gestos 100% customizados, sem a necessidade de modificar o código interno.
- 🎛️ **Dashboard Interativo**: Acompanhamento do FPS ao vivo, galeria dinâmica com os gestos habilitados, e controles integrados de qualidade de vídeo para conexões lentas.

---

## 🛠️ Como Utilizar e Treinar a Sua Própria IA

A lógica da aplicação permite a constante expansão da base de reconhecimento de gestos:

1. **Gravar Novos Gestos (`collect_landmarks.py`)**  
   Este script inicia uma interface com a câmera local exclusivamente para coletar coordenadas manuais. 
   Ao pressionar na tecla `n` na janela, você cadastra a *label* do gesto, e ao pressionar `r`, o script grava ininterruptamente todas as posições da mão para formar o banco de dados.
   Os dados são arquivados de forma bruta no `dataset/gestures.csv`.

2. **Treinar o Modelo (`train_model.py`)**  
   Com a base de dados (`gestures.csv`) preenchida com as coordenadas recém-capturadas, o script treina um modelo de classificação `Random Forest`. O script empacota seu aprendizado nos arquivos base `.joblib` na própria pasta `/models` para consumo imediato.

3. **Inspecionar no Dashboard Web (`app.py`)**  
   Subir o servidor WebSocket ativará o monitor interativo na Web. Ele consome o arquivo de IA consolidado criado no Passo 2 e projeta uma cópia em tempo real do stream classificando qualquer mão visível.

---

## 🚀 Como Executar o Projeto

### Pré-requisitos
- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** (O instalador e gestor ultrarrápido de pacotes Python da Astral)

### Execução Limpa

```bash
# 1. Acesse a raiz da aplicação
cd computer_vision_app

# 2. Reconstrua e instale todas as dependências Python
uv sync

# 3. Certifique-se de baixar e inserir a Rede Neural Base:
Verifique se existe: models/gesture_recognizer.task
>>Download: https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task

# 4. Inicie o Servidor Web e os WebSockets:
uv run app.py
```
O servidor estará acessível ao navegador em: `http://localhost:5001`.

---

## 📂 Arquitetura do Software

```text
computer_vision_app/
├── app.py              # Inicialização do FastHTML Web App e rotas /ws WebSocket
├── collect_landmarks.py# Utilitário CLI com OpenCV+Tkinter para gravar marcos.
├── train_model.py      # Lógica de ML para consumir o CSV e treinar o classificador
├── core/
│   └── processor.py    # O núcleo do motor combinando MediaPipe + Previsor Customizado
├── dataset/
│   └── gestures.csv    # Banco de Dados de coordenadas XYZ mapeadas para treinamento
├── models/
│   ├── gesture_model.joblib # O "Cérebro" Random Forest do ML
│   ├── label_encoder.joblib # Dicionário Encoder Int <-> String das categorias
│   └── gesture_recognizer.task # O extrator base de esqueleto humano TFLite do Google
└── assets/
    └── style.css / script.js / images/ # Dependências Estáticas do FastHTML
```

## HanggingFace settings
---
title: AI SignRecognizer
emoji: 🖐️
colorFrom: indigo
colorTo: green
sdk: docker
pinned: false
---

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:36:45 2026

@author: walter
"""

#Red para entrenar con los datos importados del archivo generardatos

import time
import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, Subset, TensorDataset 
from sklearn.model_selection import KFold 
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt 
import os

# Análisis de imágenes que se repiten 
def analizar_imagenes_repetidas(X):
    unicas, conteos = np.unique(X, axis=0, return_counts=True)
    total_muestras = len(X)
    total_unicas = len(unicas)
    repetidas = total_muestras - total_unicas
    
    print("\n--- ANÁLISIS DE IMÁGENES REPETIDAS ---")
    print(f"Total de imágenes cargadas: {total_muestras}")
    print(f"Imágenes con patrones únicos: {total_unicas}")
    print(f"Imágenes repetidas (duplicadas): {repetidas}")
    print(f"Patrón más frecuente repetido: {conteos.max()} veces")
    print("---------------------------------------\n")

# Modelo Red Neuronal Simple (MLP)
class MLP(nn.Module):
    def __init__(self, sizes): 
        super().__init__() 
        self.layers = nn.ModuleList() 
        for i in range(len(sizes)-1): 
            self.layers.append(nn.Linear(sizes[i], sizes[i+1])) 

    def forward(self, x): 
        h = x 
        for hidden in self.layers[:-1]: 
            h = torch.sigmoid(hidden(h)) 
        output = self.layers[-1] 
        y = output(h) 
        return y 

@torch.no_grad() 
def calcular_accuracy(predicciones, etiquetas): 
    pred_clase = predicciones.argmax(dim=1) 
    return (pred_clase == etiquetas).float().mean().item() 

# Entrenador con Gráfico: Accuracy vs K-Folds
class MLPTrainer:
    def __init__(self, datos_lista, k_folds, batch_number, epochs, repeticiones, lr=0.01):
        self.k_folds = k_folds
        self.batch_number = batch_number
        self.epochs = epochs
        self.repeticiones = repeticiones
        self.lr = lr
        
        X_np, y_np = datos_lista
        self.dataset = TensorDataset(
            torch.from_numpy(X_np).float(), 
            torch.from_numpy(y_np).long()
        )
        
        total_samples = len(self.dataset)
        samples_per_fold = total_samples / self.k_folds 
        self.batch_size = max(1, int((total_samples - samples_per_fold) / self.batch_number)) 
        
        self.acc_por_fold = np.zeros((self.repeticiones, self.k_folds))
        self.criterion = nn.MSELoss() 

    def entrenar(self):
        for r in range(self.repeticiones):
            semilla_actual = 42 + r
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=semilla_actual) 

            for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(self.dataset)))): 
                print(f"\n--- Repetición {r+1}/{self.repeticiones} | Fold {fold+1}/{self.k_folds} ---")
                
                model = MLP([16, 4]) 
                optimizer = optim.Adam(model.parameters(), lr=self.lr) 

                train_subsampler = Subset(self.dataset, train_idx) 
                val_subsampler = Subset(self.dataset, val_idx) 

                train_loader = DataLoader(train_subsampler, batch_size=self.batch_size, shuffle=True) 
                val_loader = DataLoader(val_subsampler, batch_size=len(val_subsampler), shuffle=False)  

                val_acc = 0.0
                for epoch in range(self.epochs):
                    # ===== TRAIN =====
                    model.train()
                    train_loss = 0.0
                    for x_batch, y_batch in train_loader:
                        optimizer.zero_grad() 
                        logits = model(x_batch) 
                        
                        probs = F.softmax(logits, dim=1)
                        y_one_hot = F.one_hot(y_batch, num_classes=4).float()
                        
                        loss = self.criterion(probs, y_one_hot) 
                        loss.backward() 
                        optimizer.step() 
                        train_loss += loss.item() 

                    train_loss /= len(train_loader)

                    # ===== VALIDATION =====
                    model.eval()
                    val_acc = 0.0
                    with torch.no_grad():
                        for x_val, y_val in val_loader:
                            logits_val = model(x_val)
                            probs_val = F.softmax(logits_val, dim=1)
                            val_acc += calcular_accuracy(probs_val, y_val) 

                    val_acc /= len(val_loader)
                    print(f"Época {epoch+1:02d}/{self.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}") 
                
                self.acc_por_fold[r, fold] = val_acc

        self._generar_grafico_kfolds()

    def _generar_grafico_kfolds(self):
        folds_eje = np.arange(1, self.k_folds + 1)
        
        plt.figure(figsize=(9, 5))

        for r in range(self.repeticiones):
            plt.plot(folds_eje, self.acc_por_fold[r], marker='o', linestyle='--', alpha=0.6, label=f"Repetición {r+1}")

        if self.repeticiones > 1:
            promedio_folds = np.mean(self.acc_por_fold, axis=0)
            plt.plot(folds_eje, promedio_folds, marker='s', color='black', linewidth=2.5, label="Promedio por Fold")

        plt.title("Accuracy de Validación por Fold")
        plt.xlabel("Folds (K-Fold)")
        plt.ylabel("Accuracy Final")
        plt.xticks(folds_eje)
        plt.grid(True, linestyle=":", alpha=0.7)
        plt.legend()

        plt.tight_layout()
        plt.savefig("accuracy_vs_kfolds.png", dpi=300)
        print("\n[INFO] Gráfico guardado como 'accuracy_vs_kfolds.png'")
        plt.show()

# PROGRAMA PRINCIPAL
# PROGRAMA PRINCIPAL
def main():
    tiempo_inicio = time.time()
    
    print("=== INICIANDO EXPERIMENTO SINTÉTICO (4x4) ===")
    
    # 1. Cargar el dataset desde el archivo externo usando la ruta absoluta segura
    ruta_dataset = os.path.join(os.path.dirname(__file__), "dataset_sintetico.npz")
    print(f"Cargando dataset desde: {ruta_dataset}")
    
    if not os.path.exists(ruta_dataset):
        print(f"\n[ERROR] No se encuentra el archivo 'dataset_sintetico.npz'.")
        print("Por favor, ejecuta primero el script 'generar_datos.py' para crearlo.")
        return

    data = np.load(ruta_dataset)
    X, y = data['X'], data['y']
    
    # 2. Análisis de imágenes repetidas cargadas
    analizar_imagenes_repetidas(X)
    
    K_FOLDS = 50             
    BATCH_NUMBER = 1        
    EPOCHS = 15             
    REPETICIONES = 1        
    LEARNING_RATE = 0.01    
    
    # 3. Entrenar y graficar
    trainer = MLPTrainer(
        datos_lista=[X, y], 
        k_folds=K_FOLDS, 
        batch_number=BATCH_NUMBER, 
        epochs=EPOCHS, 
        repeticiones=REPETICIONES,
        lr=LEARNING_RATE
    )
    trainer.entrenar()
    
    tiempo_total = time.time() - tiempo_inicio
    print(f"\n=== ¡PROCESO TERMINADO CON ÉXITO! ===")
    print(f"Tiempo total de ejecución: {tiempo_total:.2f} segundos")

if __name__ == "__main__":
    main()
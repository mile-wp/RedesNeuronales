import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, Subset, TensorDataset 
from sklearn.model_selection import KFold 
import torch.nn.functional as F 
import numpy as np 
import matplotlib.pyplot as plt 
import os 
import torchvision 
import torchvision.transforms as transforms 
import time 
import mlflow

class MLP(torch.nn.Module):
    def __init__(self, sizes): 
        super().__init__() 
        self.layers = torch.nn.ModuleList() 
        for i in range(len(sizes)-1): 
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i+1])) 

    def forward(self, x): 
        h = x 
        for hidden in self.layers[:-1]: 
            # Usamos sigmoid para evitar warnings de softmax en capas ocultas intermedias
            h = torch.sigmoid(hidden(h)) 
        output = self.layers[-1] 
        y = output(h) 
        return y 


@torch.no_grad() 
def calcular_accuracy(predicciones, etiquetas): 
    pred_clase = predicciones.argmax(dim=1) 
    return (pred_clase == etiquetas).float().mean().item() 


class MLPTrainer:
    def __init__(self, datos_lista, k_folds, batch_number, epochs, repeticiones):
        self.k_folds = k_folds
        self.batch_number = batch_number
        self.epochs = epochs
        self.repeticiones = repeticiones
        
        X_np, y_np = datos_lista
        self.mnist_dataset = TensorDataset(
            torch.from_numpy(X_np).float(), 
            torch.from_numpy(y_np).long()
        )
        
        total_images = len(self.mnist_dataset)
        images_per_fold = total_images / self.k_folds 
        self.batch_size = max(1, int((total_images - images_per_fold) / self.batch_number)) 
        
        self.total_ejecuciones = self.repeticiones * self.k_folds
        self.curvas_historial = np.empty(self.total_ejecuciones, dtype=object) 
        
        self.criterion = nn.MSELoss() 
        
        # --- MLFLOW: Nombre del experimento ---
        mlflow.set_experiment("Mi_Prueba_Rapida_MLflow")

    def entrenar(self):
        with mlflow.start_run(run_name="Run_Rapido") as run_padre:
            
            mlflow.log_param("k_folds", self.k_folds)
            mlflow.log_param("epochs", self.epochs)
            mlflow.log_param("batch_size", self.batch_size)

            for r in range(self.repeticiones):
                semilla_actual = 42 + r
                kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=semilla_actual) 

                for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(self.mnist_dataset)))): 
                    print(f"\n--- Ejecución {r+1} | Fold {fold+1}/{self.k_folds} ---")
                    
                    with mlflow.start_run(run_name=f"Rep_{r+1}_Fold_{fold+1}", nested=True):
                        
                        model = MLP([28*28, 10]) 
                        optimizer = optim.Adam(model.parameters(), lr=0.01) 

                        train_subsampler = Subset(self.mnist_dataset, train_idx) 
                        val_subsampler = Subset(self.mnist_dataset, val_idx) 

                        train_loader = DataLoader(train_subsampler, batch_size=self.batch_size, shuffle=True) 
                        val_loader = DataLoader(val_subsampler, batch_size=len(val_subsampler), shuffle=False)  

                        val_acc_historial = []

                        for epoch in range(self.epochs):
                            # ===== TRAIN =====
                            model.train()
                            train_loss = 0.0
                            for x_batch, y_batch in train_loader:
                                optimizer.zero_grad() 
                                logits = model(x_batch) 
                                y_one_hot = F.one_hot(y_batch, num_classes=10).float()
                                
                                loss = self.criterion(logits, y_one_hot) 
                                loss.backward() 
                                optimizer.step() 
                                train_loss += loss.item() 

                            train_loss /= len(train_loader)

                            # ===== VALIDATION =====
                            model.eval()
                            val_acc = 0.0
                            with torch.no_grad():
                                for x_val, y_val in val_loader:
                                    preds = model(x_val) 
                                    val_acc += calcular_accuracy(preds, y_val) 

                            val_acc /= len(val_loader)
                            val_acc_historial.append(val_acc)

                            # --- MLFLOW: Solo guardamos las métricas en texto (súper liviano) ---
                            mlflow.log_metric(f"loss_fold_{fold+1}", train_loss, step=epoch)
                            mlflow.log_metric(f"accuracy_fold_{fold+1}", val_acc, step=epoch)

                            print(f"Epoch {epoch+1:02d}/{self.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}") 
                        
                        indice_fila = fold + (r * self.k_folds) 
                        self.curvas_historial[indice_fila] = np.array(val_acc_historial)
                        
                        # Quité mlflow.pytorch.log_model para evitar problemas de dependencias y guardar archivos pesados
                        
        return self.curvas_historial

    def graficar(self):
        plt.figure(figsize=(10, 6))
        for i in range(self.total_ejecuciones):
            plt.plot(self.curvas_historial[i], alpha=0.6) 
        plt.xlabel("Epochs") 
        plt.ylabel("Accuracy") 
        plt.title(f"Prueba Rápida - Perceptrón Único")
        plt.grid(True)
        
        plt.savefig("grafico_saturacion.png")
        mlflow.log_artifact("grafico_saturacion.png")
        plt.close() # Cierra el gráfico para que no bloquee la consola


def cargar_datos_lista_numpy():
    # Usamos solo una fracción diminuta del dataset original para que corra en segundos en CPU
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    
    # Reducimos los datos a solo 1000 imágenes para la prueba ultra-rápida
    X = dataset.data[:1000].reshape(-1, 28*28).float() / 255.0 
    y = dataset.targets[:1000]
    return [X.numpy(), y.numpy()]


# PROGRAMA PRINCIPAL
def main():
    print("=== INICIANDO PRUEBA RAPIDA ===")
    datos_entrenamiento = cargar_datos_lista_numpy()
    
    # Parámetros súper livianos
    k_folds = 3 
    batch_number = 10  # Batch size cómodo de ~66 imágenes
    epochs = 5         # Solo 5 épocas, termina al toque
    repeticiones = 1

    trainer = MLPTrainer(datos_entrenamiento, k_folds, batch_number, epochs, repeticiones)
    trainer.entrenar()
    trainer.graficar()
    print("\n=== ¡PROCESO TERMINADO CON ÉXITO! ===")

if __name__ == "__main__":
    main()
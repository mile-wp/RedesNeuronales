
""""

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

class MLP(torch.nn.Module):
    def __init__(self, sizes): 
        super().__init__() 
        self.layers = torch.nn.ModuleList() 
        for i in range(len(sizes)-1): 
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i+1])) 

    def forward(self, x): 
        h = x 
        for hidden in self.layers[:-1]: 
            h = F.relu(hidden(h)) 
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
        
        # --- PROCESAMIENTO DE LA LISTA DE NUMPY ARRAYS ---
        # datos_lista esperado: [X_train_numpy, y_train_numpy]
        X_np, y_np = datos_lista
        self.mnist_dataset = TensorDataset(
            torch.from_numpy(X_np).float(), 
            torch.from_numpy(y_np).long()
        )
        
        total_images = len(self.mnist_dataset)
        images_per_fold = total_images / self.k_folds 
        self.batch_size = max(1, int((total_images - images_per_fold) / self.batch_number)) 
        
        # Cada curva completa será un elemento dentro de este array objeto de NumPy
        self.total_ejecuciones = self.repeticiones * self.k_folds
        self.curvas_historial = np.empty(self.total_ejecuciones, dtype=object) 
        
        # Función de costo configurada para MSE
        self.criterion = nn.MSELoss() 

    def entrenar(self):
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42) 

        for r in range(self.repeticiones):
            for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(self.mnist_dataset)))): 
                print(f"\n--- Ejecución {r+1} | Fold {fold+1}/{self.k_folds} ---")
                
                model = MLP([28*28, 10]) 
                optimizer = optim.Adam(model.parameters(), lr=0.001) 

                train_subsampler = Subset(self.mnist_dataset, train_idx) 
                val_subsampler = Subset(self.mnist_dataset, val_idx) 

                train_loader = DataLoader(train_subsampler, batch_size=self.batch_size, shuffle=True) 
                val_loader = DataLoader(val_subsampler, batch_size=len(val_subsampler), shuffle=False)  

                val_acc_historial = []

                for epoch in range(self.epochs):
                    inicio_epoch = time.time() 

                    # ===== TRAIN =====
                    model.train()
                    train_loss = 0.0

                    for x_batch, y_batch in train_loader:
                        optimizer.zero_grad() 
                        logits = model(x_batch) 
                        
                        # Adaptación requerida para MSE: Convertir etiquetas targets a vectores One-Hot flotantes
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

                    tiempo_epoch = time.time() - inicio_epoch 

                    if (epoch + 1) % 5 == 0 or epoch == 0:
                        print(f"Epoch {epoch+1:03d}/{self.epochs} | Loss (MSE): {train_loss:.4f} | Val Acc: {val_acc:.4f} | Tiempo: {tiempo_epoch:.2f}s") 
                
                # Almacenamos la curva como elemento del array
                indice_fila = fold + (r * self.k_folds) 
                self.curvas_historial[indice_fila] = np.array(val_acc_historial)
                        
        return self.curvas_historial

    def graficar(self):
        plt.figure(figsize=(10, 6))
        for i in range(self.total_ejecuciones):
            plt.plot(self.curvas_historial[i], label=f"Fold {i+1}") 

        plt.xlabel("epochs") 
        plt.ylabel("accuracy") 
        plt.title(f"Curvas de Validación - {self.k_folds} Kfolds (MSE Loss)")
        plt.grid(True)
        if self.k_folds <= 10:  
            plt.legend()
        plt.show()


def cargar_datos_lista_numpy():
    #Descarga los datos y los empaqueta estrictamente en una lista de arrays de NumPy
    
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    
    ya_descargado = os.path.exists('./data/FashionMNIST')
    debe_descargar = not ya_descargado

    dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=debe_descargar, transform=transform
    )
    
    X = dataset.data.reshape(-1, 28*28).float() / 255.0 
    y = dataset.targets 
    
    # Retornamos la lista con arrays de NumPy independientes para X e y
    print("[INFO] Datos cargados exitosamente como una lista de NumPy arrays.")
    return [X.numpy(), y.numpy()]


# PROGRAMA PRINCIPAL AUTOMATIZADO (Sin inputs de consola)
def main():
    print("=== INICIANDO PIPELINE DE ENTRENAMIENTO AUTOMÁTICO ===")
    
    # 1. Carga de datos directamente como una lista [X_numpy, y_numpy]
    datos_entrenamiento = cargar_datos_lista_numpy()
    
    # 2. Configuración predefinida directamente en variables de código (sin prompts)
    k_folds = 50
    batch_number = 1
    epochs = 25  # Configurado a 25 para ver reportes rápidos cada 5 épocas
    repeticiones = 1

    print("----------------------------------------------------")
    print(f"Configuración establecida -> Kfolds: {k_folds}, Batches: {batch_number}, Epochs: {epochs}")
    print("----------------------------------------------------")
    
    # 3. Inicializar ejecutor pasándole la lista NumPy y las variables
    trainer = MLPTrainer(datos_entrenamiento, k_folds, batch_number, epochs, repeticiones)
    trainer.entrenar()
    
    print("\nEntrenamiento finalizado con éxito. Generando gráficos...")
    trainer.graficar()

if __name__ == "__main__":
    main()

    """

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

class MLP(torch.nn.Module):
    def __init__(self, sizes): 
        super().__init__() 
        self.layers = torch.nn.ModuleList() 
        for i in range(len(sizes)-1): 
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i+1])) 

    def forward(self, x): 
        h = x 
        for hidden in self.layers[:-1]: 
            # --- CAMBIO REALIZADO: Se reemplazó F.relu por F.softmax ---
            h = F.softmax(hidden(h), dim=1) 
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
        
        # --- PROCESAMIENTO DE LA LISTA DE NUMPY ARRAYS ---
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

    def entrenar(self):
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42) 

        for r in range(self.repeticiones):
            for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(len(self.mnist_dataset)))): 
                print(f"\n--- Ejecución {r+1} | Fold {fold+1}/{self.k_folds} ---")
                
                model = MLP([28*28, 10]) 
                optimizer = optim.Adam(model.parameters(), lr=0.001) 

                train_subsampler = Subset(self.mnist_dataset, train_idx) 
                val_subsampler = Subset(self.mnist_dataset, val_idx) 

                train_loader = DataLoader(train_subsampler, batch_size=self.batch_size, shuffle=True) 
                val_loader = DataLoader(val_subsampler, batch_size=len(val_subsampler), shuffle=False)  

                val_acc_historial = []

                for epoch in range(self.epochs):
                    inicio_epoch = time.time() 

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

                    tiempo_epoch = time.time() - inicio_epoch 

                    if (epoch + 1) % 5 == 0 or epoch == 0:
                        print(f"Epoch {epoch+1:03d}/{self.epochs} | Loss (MSE): {train_loss:.4f} | Val Acc: {val_acc:.4f} | Tiempo: {tiempo_epoch:.2f}s") 
                
                indice_fila = fold + (r * self.k_folds) 
                self.curvas_historial[indice_fila] = np.array(val_acc_historial)
                        
        return self.curvas_historial

    def graficar(self):
        plt.figure(figsize=(10, 6))
        for i in range(self.total_ejecuciones):
            plt.plot(self.curvas_historial[i], label=f"Fold {i+1}") 

        plt.xlabel("epochs") 
        plt.ylabel("accuracy") 
        plt.title(f"Curvas de Validación - {self.k_folds} Kfolds (MSE Loss con Softmax)")
        plt.grid(True)
        if self.k_folds <= 10:  
            plt.legend()
        plt.show()


def cargar_datos_lista_numpy():
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    
    ya_descargado = os.path.exists('./data/FashionMNIST')
    debe_descargar = not ya_descargado

    dataset = torchvision.datasets.FashionMNIST(
        root='./data', train=True, download=debe_descargar, transform=transform
    )
    
    X = dataset.data.reshape(-1, 28*28).float() / 255.0 
    y = dataset.targets 
    
    print("[INFO] Datos cargados exitosamente como una lista de NumPy arrays.")
    return [X.numpy(), y.numpy()]


# PROGRAMA PRINCIPAL AUTOMATIZADO
def main():
    print("=== INICIANDO PIPELINE DE ENTRENAMIENTO AUTOMÁTICO ===")
    
    datos_entrenamiento = cargar_datos_lista_numpy()
    
    k_folds = 50
    batch_number = 1
    epochs = 25  
    repeticiones = 1

    print("----------------------------------------------------")
    print(f"Configuración establecida -> Kfolds: {k_folds}, Batches: {batch_number}, Epochs: {epochs}")
    print("----------------------------------------------------")
    
    trainer = MLPTrainer(datos_entrenamiento, k_folds, batch_number, epochs, repeticiones)
    
    # --- CAMBIO REALIZADO: Medición del tiempo total de la red ---
    inicio_total = time.time()
    
    trainer.entrenar()
    
    tiempo_total = time.time() - inicio_total
    # -------------------------------------------------------------
    
    print("\n====================================================")
    print(f" TIEMPO TOTAL DE EJECUCIÓN: {tiempo_total:.2f} segundos ({tiempo_total/60:.2f} minutos)")
    print("====================================================")
    
    print("\nGenerando gráficos...")
    trainer.graficar()

if __name__ == "__main__":
    main()
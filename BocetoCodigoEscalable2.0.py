                    # BOCETO DE NUEVAS IMPLEMENTACIONES PARA HACERLO MÁS ESCALABLE

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

import time #Para usar de contador del tiempo que tarda cada epoch


class MLP(torch.nn.Module):
    def __init__(self, sizes): 
        super().__init__() 
        self.layers = torch.nn.ModuleList() 
        for i in range(len(sizes)-1): 
            self.layers.append(torch.nn.Linear(sizes[i], sizes[i+1])) 

    def forward(self, x): 
        h = x 
        for hidden in self.layers[:-1]: 
            h = F.relu(hidden(h)) #Implementación futura-> que permita cambiar la función
        output = self.layers[-1] 
        y = output(h) 
        return y 


@torch.no_grad() 
def calcular_accuracy(predicciones, etiquetas): 
    pred_clase = predicciones.argmax(dim=1) 
    return (pred_clase == etiquetas).float().mean().item() 


# NUEVA CLASE: MLPTrainer (Encapsula el pipeline de entrenamiento)
class MLPTrainer:
    def __init__(self, dataset, k_folds, batch_number, epochs, repeticiones):
        self.mnist_dataset = dataset
        self.k_folds = k_folds
        self.batch_number = batch_number
        self.epochs = epochs
        self.repeticiones = repeticiones
        
        # Ajuste dinámico de tamaños según k_folds
        # Usamos el total de datos del dataset recibido para calcular los batches
        total_images = len(self.mnist_dataset)
        images_per_fold = total_images / self.k_folds #IMPORTANTE mejora futura-> si el usuario ingresa por ejemplo 100, se rompe el código
        self.batch_size = int((total_images - images_per_fold) / self.batch_number) # CONSULTAR--> Al usar total_imagenes hace que sea muy pesado. 
                                                                                    # Posible cambio: self.batch_size = batch_size y usar uno predeterminado ej 64 o que se ingrese por teclado
        
        # Inicialización de la matriz de almacenamiento de accuracy
        self.acc = np.zeros([self.repeticiones * self.k_folds, self.epochs]) 
        self.criterion = nn.CrossEntropyLoss() #Función de costo

#Ejecuta el loop de entrenamiento por folds y epochs"
    def entrenar(self):
        
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42) 

        for r in range(self.repeticiones):
            for fold, (train_idx, val_idx) in enumerate(kf.split(self.mnist_dataset)): 
                print(f"\n--- Ejecución {r+1} | Fold {fold+1}/{self.k_folds} ---")
                
                
                model = MLP([28*28, 10]) #Mejora futura-> ahora como está adentro del for cada vez que itera (cada fold que pasa) se borra lo anterior entonces podríamos ver una manera en la cual de alguna manera se guarden los datos previos
                optimizer = optim.Adam(model.parameters(), lr=0.001) 

                # Subsets para el fold actual--> en vez de copiar las imagenes las visita. Es como una vista a las imagenes
                train_subsampler = Subset(self.mnist_dataset, train_idx) 
                val_subsampler = Subset(self.mnist_dataset, val_idx) 


                #Ahora dataloader solo procesa datos que fueron filtrados por subset-> CONSULTAR
                train_loader = DataLoader(train_subsampler, batch_size=self.batch_size, shuffle=True) #Para training
                val_loader = DataLoader(val_subsampler, batch_size=len(val_subsampler), shuffle=False)  #Para validation


                for epoch in range(self.epochs):
                    inicio_epoch = time.time() # CAMBIO Iniciamos el cronómetro


                    # ===== TRAIN =====
                    model.train()
                    train_loss = 0.0

                    for x_batch, y_batch in train_loader:
                        optimizer.zero_grad() 
                        logits = model(x_batch) 
                        loss = self.criterion(logits, y_batch) 
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
                    
                    # Guardado indexado dinámicamente
                    indice_fila = fold + (r * self.k_folds) #En vez del 5 ponemos self para que pueda funcionar con cualquier cantidad de kfolds que le settemos
                    self.acc[indice_fila, epoch] = val_acc 


                    tiempo_epoch = time.time() - inicio_epoch #CAMBIO


                    
                    # En vez de imprimir todos los epochs va imprimiendo periodicamente para que no se sature la consola-> CONSULTAR
                    if (epoch + 1) % 25 == 0 or epoch == 0:
                        #Para que quede más prolijo por consola 
                        print(f"Epoch {epoch+1:03d}/{self.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Tiempo: {tiempo_epoch:.2f}s") #Agregamos el cronómetro
                        
        return self.acc

#Genera el gráfico final de las curvas obtenidas-> CONSULTAR: ¿Es mejor esto o dejamos matplotlib?

    def graficar(self):
        
        plt.figure(figsize=(10, 6))
        for i in range(self.repeticiones * self.k_folds):
            plt.plot(self.acc[i, :], label=f"Fold {i+1}") 

        plt.xlabel("epochs") 
        plt.ylabel("accuracy") 
        plt.title(f"Curvas de Validación - {self.k_folds} Kfolds")
        plt.grid(True)
        if self.k_folds <= 10:  # Hace que se pueda graficar independientemente de que tan grande sea ,evita romper el gráfico
            plt.legend()
        plt.show()


# Carga eficiente mediante bucle--> evitamos que se descargue si no es necesario
""" 
Descarga e inicializa el dataset usando un bucle para evitar duplicar código.
Verifica si los datos ya existen localmente en './data' antes de llamar a download.
"""
def cargar_datos_eficiente(usar_entrenamiento=True):
    
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    
    # Determinar si descargamos o no según la existencia de la carpeta
    ya_descargado = os.path.exists('./data/FashionMNIST')
    debe_descargar = not ya_descargado

    datasets_cargados = {}
    
    # Unificamos las dos opciones (train = true y train = false)
    # Si usar_entrenamiento es True procesa ambos, sino solo procesa Test (False)
    modos_a_cargar = [True, False] if usar_entrenamiento else [False]
    
    for modo in modos_a_cargar:
        dataset = torchvision.datasets.FashionMNIST(
            root='./data',
            train=modo,
            download=debe_descargar,
            transform=transform
        )
        
        # Aplana el tamaño de las imágenes
        X = dataset.data.reshape(-1, 28*28).float() / 255.0 #toma la grilla de cada imagen y estira sus filas una detrás de otra hasta transformarla en una única línea continua de 784 píxeles
        y = dataset.targets #etiquetas reales asociadas a cada imagen
        
        key = 'train' if modo else 'test'
        datasets_cargados[key] = TensorDataset(X, y)
        
    # Retorna el TensorDataset solicitado
    if usar_entrenamiento:
        print("Utilizando datos de ENTRENAMIENTO (FashionMNIST Train).")
        return datasets_cargados['train']
    else:
        print("Utilizando datos de PRUEBA (FashionMNIST Test).")
        return datasets_cargados['test']



# PROGRAMA PRINCIPAL
def solicitar_parametro(mensaje, valor_defecto):
    #Función auxiliar para consultar valores manteniendo el predeterminado
    respuesta = input(f"{mensaje} [Por defecto: {valor_defecto}]: ").strip()
    if respuesta == "":
        return valor_defecto
    return int(respuesta)


#Mini menú para permitir cambios
def main():
    print("    ---  MENU PRINCIPAL   ---    ")
    
    # Preguntar qué datos utilizar utilizando el nuevo flujo eficiente
    resp_datos = input("¿Desea usar los datos de entrenamiento? (S/N) [Por defecto: S]: ").strip().upper()
    usar_entrenamiento = False if resp_datos == "N" else True
    
    # Carga eficiente del dataset (Sin variables globales)
    mnist_dataset = cargar_datos_eficiente(usar_entrenamiento)
    
    # Valores que setteamos por default
    k_folds = 5
    batch_number = 1
    epochs = 100
    repeticiones = 1

    # Confirmación de valores
    print("\n----------------------------------------------------")
    print("Valores predeterminados actuales:")
    print(f"  - Kfolds: {k_folds}")
    print(f"  - Mini-batches: {batch_number}")
    print(f"  - Épocas por fold: {epochs}")
    print(f"  - Repeticiones del experimento: {repeticiones}")
    print("----------------------------------------------------")
    
    confirmacion = input("¿Desea conservar los siguientes valores? (S/N) [Por defecto: S]: ").strip().upper()

    if confirmacion == "N":
        # Si dice que no, permitimos que cambie las variables una por una
        print("\n[MODIFICACIÓN DE PARÁMETROS]")
        k_folds = solicitar_parametro("Ingrese cantidad de Kfolds (Sugerido 5 a 20)", k_folds)
        batch_number = solicitar_parametro("Ingrese número de mini-batches", batch_number)
        epochs = solicitar_parametro("Ingrese cantidad de epochs por fold", epochs)
        repeticiones = solicitar_parametro("Ingrese cantidad de repeticiones del experimento", repeticiones)
    else:
        # Si dice que sí (o presiona Enter por defecto)
        print("\nCorriendo programa...")

    # Validaciones rápidas de seguridad
    if k_folds < 2:
        print("Error: Kfolds mínimo debe ser 2. Seteando k_folds = 2.")
        k_folds = 2

    print("\n[INFO] Iniciando Pipeline de Entrenamiento...")
    print(f"Configuración final -> Kfolds: {k_folds}, Batches: {batch_number}, Epochs: {epochs}, Repeticiones: {repeticiones}")
    
    # Instanciar la clase controladora y correr el proceso
    trainer = MLPTrainer(mnist_dataset, k_folds, batch_number, epochs, repeticiones)
    trainer.entrenar()
    
    print("\nEntrenamiento finalizado con éxito. Generando gráficos...")
    trainer.graficar()

if __name__ == "__main__":
    main()
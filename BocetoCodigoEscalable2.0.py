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
            h = F.relu(hidden(h))  #Implementación futura-> que permita cambiar la función
        output = self.layers[-1] 
        y = output(h) 
        return y 


@torch.no_grad() 
def calcular_accuracy(predicciones, etiquetas): 
    pred_clase = predicciones.argmax(dim=1) 
    return (pred_clase == etiquetas).float().mean().item() 




#  Funciones para calcular media y dispersión
def metrica_promedio_final(curva):
    """Calcula el promedio de los últimos 5 epochs (convergencia)
    Mejora futura: Que no sea predeterminada la cantidad de epochs
    """
    return np.mean(curva[-5:]) if len(curva) >= 5 else np.mean(curva)

def metrica_pico_maximo(curva):
    """Devuelve la precisión máxima alcanzada en el fold."""
    return np.max(curva)



# NUEVA CLASE: MLPTrainer (Encapsula el pipeline de entrenamiento)

class MLPTrainer:
    def __init__(self, datos_lista, k_folds, batch_number, epochs, repeticiones):
        self.k_folds = k_folds
        self.batch_number = batch_number
        self.epochs = epochs
        self.repeticiones = repeticiones
        
        # NUEVO: ADMITE PASAR UNA LISTA CON LOS ARRAYS DE ENTRENAMIENTO [X, y] 
        if isinstance(datos_lista, list) and len(datos_lista) == 2: #Verifica que tipo de dato tiene la variable.
            
            #Si es true -> entonces pasó la lista y ahí entra en el if, separa los datos y convierte a tensores de pytorch. Extraemos los datos de la lista y los transformamos en el TensorDataset que PyTorch necesita

            #Si es false -> entiende que no le pasamos una lista, sino un objeto tipo tensordataset entonces dice "listo uso esto, sin conversión"

            #Esto sería el true
            X_np, y_np = datos_lista
            self.mnist_dataset = TensorDataset(
                torch.from_numpy(X_np).float(), 
                torch.from_numpy(y_np).long()
            )

            #Esto sería el false
        else:
            self.mnist_dataset = datos_lista
        
        
       # Usamos el total de datos del dataset recibido para calcular los batches

        total_images = len(self.mnist_dataset)
        images_per_fold = total_images / self.k_folds 
        self.batch_size = max(1, int((total_images - images_per_fold) / self.batch_number)) 
        
        # Cada curva es un elemento individual de este array de objetos
        self.total_ejecuciones = self.repeticiones * self.k_folds
        self.curvas_historial = np.empty(self.total_ejecuciones, dtype=object)
        
        self.funciones_analisis = np.array([metrica_promedio_final, metrica_pico_maximo], dtype=object) #dtype-> le decimos que todos los datos que se van a guardar van a ser genéricos.
       
        #Por ej si hubieramos puesto que se guarden solo números se hubiera roto porque acá le estamos mandando funciones
        
        self.criterion = nn.CrossEntropyLoss() #Función de costo

    #Empieza a entrenar
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

                # Se va armando la curva para el fold que está pasando en ese momento específico y se va guardando en el historial para el json
    
                val_acc_historial = []

                for epoch in range(self.epochs):
                    inicio_epoch = time.time() 

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
                    val_acc_historial.append(val_acc)

                    tiempo_epoch = time.time() - inicio_epoch 

                    if (epoch + 1) % 25 == 0 or epoch == 0:
                        print(f"Epoch {epoch+1:03d}/{self.epochs} | Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Tiempo: {tiempo_epoch:.2f}s") 
                
                # Cada curva completa se guarda acá para el json
                indice_fila = fold + (r * self.k_folds) #En vez del 5 ponemos self para que pueda funcionar con cualquier cantidad de kfolds que le settemos
                self.curvas_historial[indice_fila] = np.array(val_acc_historial)

        #Usa self porque llama a otra función adentro de la misma clase
        self.ejecutar_analisis_post_entrenamiento() #primero pausa el método entrenar y después salta a la función ejecutar_analisis_post_entrenamiento(self), ejecuta la función
        
        #después cuando termina vuelve acá y pasa a la siguiente línea
        return self.curvas_historial #En este línea (después de haber pasado por ejecutar análisis ...) básicamente se guarda todo lo que hizo en el historial (json)


    #Tomar con pinzas!

    def ejecutar_analisis_post_entrenamiento(self):
        print("ANÁLISIS DE RESULTADOS USANDO LAS FUNCIONES GENERADAS")
        for i, curva in enumerate(self.curvas_historial):
            print(f"\n> Análisis de la Curva {i+1} (Elemento del array de NumPy):")
            for func in self.funciones_analisis:
                nombre_metrica = func.__name__ #Agarra el nombre de la función----> a medida que va iterando va agarrando las funciones que están, en orden
                #En la primera iteración agarra a metrica_promedio_final y calcula el promedio. En la segunda iteración agarra a metrica_pico_maximo y hace el cálculo
                #Está bueno porque si queremos agregar otra función no tenemos que cambiar tanto el código. Solamente agregarla arriba junto con las otras
                
                #Pero tal vez queda un código muy pesado y engorroso-> chequear
            
                resultado = func(curva) #Le pasa los datos a la función que agarró antes
                print(f"  - {nombre_metrica}: {resultado:.4f}") #Imprime tipo "metrica_promedio_final" : valor calculado



    def graficar(self):
        plt.figure(figsize=(10, 6))
        for i in range(self.total_ejecuciones):
            # Grafica cada curva que actúa como elemento del array principal
            plt.plot(self.curvas_historial[i], label=f"Ejecución/Fold {i+1}") 

        plt.xlabel("epochs") 
        plt.ylabel("accuracy") 
        plt.title(f"Curvas de Validación - {self.k_folds} Kfolds")
        plt.grid(True)
        if self.k_folds <= 10:  
            plt.legend()
        plt.show()


# Devuelve una lista con los arrays de numpy [X_numpy, y_numpy]
def cargar_datos_eficiente(usar_entrenamiento=True):
    transform = transforms.Compose([ 
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,)) 
    ])
    
    ya_descargado = os.path.exists('./data/FashionMNIST')
    debe_descargar = not ya_descargado
    datasets_cargados = {}
    modos_a_cargar = [True, False] if usar_entrenamiento else [False]
    
    for modo in modos_a_cargar:
        dataset = torchvision.datasets.FashionMNIST(
            root='./data', train=modo, download=debe_descargar, transform=transform
        )
        X = dataset.data.reshape(-1, 28*28).float() / 255.0 
        y = dataset.targets 
        
        key = 'train' if modo else 'test'
        # Guardamos como una lista nativa que contiene los arrays de NumPy
        datasets_cargados[key] = [X.numpy(), y.numpy()]
        
    if usar_entrenamiento:
        print("Utilizando lista de entrenamiento con NumPy arrays.")
        return datasets_cargados['train']
    else:
        print("Utilizando lista de prueba con NumPy arrays.")
        return datasets_cargados['test']


# PROGRAMA PRINCIPAL   ¡Cambio futuro-> que se automatice!

def solicitar_parametro(mensaje, valor_defecto):
    #Función auxiliar para consultar valores manteniendo el predeterminado
    respuesta = input(f"{mensaje} [Por defecto: {valor_defecto}]: ").strip()
    if respuesta == "":
        return valor_defecto
    return int(respuesta)



#mini menú
def main():
    print("  MENU PRINCIPAL ")
    
    resp_datos = input("¿Desea usar los datos de entrenamiento? (S/N) [Por defecto: S]: ").strip().upper()
    usar_entrenamiento = False if resp_datos == "N" else True
    
    # Carga los datos formateados como una lista: [X_train, y_train] con arrays de NumPy
    datos_entrenamiento_lista = cargar_datos_eficiente(usar_entrenamiento)
    
    # Valores que setteamos por default  (Cambie epoch=25 para probar)
    k_folds = 5
    batch_number = 1
    epochs = 25
    repeticiones = 1

    print("\n----------------------------------------------------")
    print("Valores predeterminados actuales:")
    print(f"  - Kfolds: {k_folds}")
    print(f"  - Mini-batches: {batch_number}")
    print(f"  - Épocas por fold: {epochs}")
    print(f"  - Repeticiones del experimento: {repeticiones}")
    print("----------------------------------------------------")
    
    confirmacion = input("¿Desea conservar los siguientes valores? (S/N) [Por defecto: S]: ").strip().upper()

    if confirmacion == "N":
        print("\n[MODIFICACIÓN DE PARÁMETROS]")
        k_folds = solicitar_parametro("Ingrese cantidad de Kfolds", k_folds)
        batch_number = solicitar_parametro("Ingrese número de mini-batches", batch_number)
        epochs = solicitar_parametro("Ingrese cantidad de epochs por fold", epochs)
        repeticiones = solicitar_parametro("Ingrese cantidad de repeticiones del experimento", repeticiones)

    if k_folds < 2:
        print("Error: Kfolds mínimo debe ser 2. Seteando k_folds = 2.")
        k_folds = 2

    print("\n[INFO] Iniciando Pipeline de Entrenamiento...")
    
    # Pasamos la lista con arrays directamente al Trainer sin usar diccionarios de configuración
    trainer = MLPTrainer(datos_entrenamiento_lista, k_folds, batch_number, epochs, repeticiones)
    trainer.entrenar()
    trainer.graficar()

if __name__ == "__main__":
    main()

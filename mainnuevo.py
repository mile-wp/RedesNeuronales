#Nuevo archivo main. Este se centra únicamente en comparar las imagenes creadas y guardarlas en el dataset. 

import os
import numpy as np


def comparar_imagenes(X_datos, imagen_nueva):
    """
    Compara la imagen nueva contra todas las imágenes cargadas del dataset.
    Devuelve:
        True --> Si es IGUAL a alguna existente
        False --> Si es DIFERENTE a todas
    """
    if len(X_datos) == 0:
        return False

    vectores_existentes = np.array([img.flatten() for img in X_datos]) #usamos flatten para "aplanar" la imagen y trabajar con vectores
    vec_nuevo = imagen_nueva.flatten()

    matriz_pixeles = (vectores_existentes == vec_nuevo)

    for idx, coincidencia_pixeles in enumerate(matriz_pixeles):
        if np.all(coincidencia_pixeles):
            print(f"  [DUPLICADO DETECTADO] Coincide con la imagen en el índice {idx}")
            print(f"  Vector Booleano: {coincidencia_pixeles.tolist()}")
            return True 

    return False

#Consultar si está bien que la probabilidad sea de 0.08
def agregar_imagen_al_dataset(ruta_archivo, clase_deseada=0, flip_probability=0.08, seed=None): #Se agrega la imagen al dataset únicamente si no hay una imagen igual cargada
    if not os.path.exists(ruta_archivo):
        print(f"[ERROR] No se encuentra el archivo '{ruta_archivo}'. Ejecutá primero el generador.") #Primero hay que correr el dataset para saber qué imagenes tiene cargadas
        return

    #Cargar el dataset guardado
    data = np.load(ruta_archivo)
    X = list(data['X'])
    y = list(data['y'])

    print(f"--- DATASET CARGADO ---")
    print(f"Total de imágenes previas: {len(X)}")

    # Tomar el patrón base directamente desde el dataset cargado. 
    # Hace que la nueva imagen se pueda generar en formato de matriz. Es decir, que en vez de que queden como vectores se reescribe como una imagen de 4x4 pixeles
    patron_base = X[clase_deseada].reshape(4, 4).astype(np.uint8)

    # Generar la nueva imagen con ruido --> consultar
    rng = np.random.default_rng(seed)

    ruido = rng.random((4, 4)) < flip_probability #Para determinar que pixel va a cambiar. True si la posición es < 0.08 y false si es mayor

    imagen_nueva = np.logical_xor(patron_base, ruido).astype(np.uint8) #Con true los pixeles cambian de estado y con false quedan iguales. Reglita xor--> 0+0 =0, 1+0=1, 1+1=0

    # Comparar y guardar si es única
    print("\n--- ANALIZANDO IMAGEN NUEVA ---")
    es_repetida = comparar_imagenes(X_datos=X, imagen_nueva=imagen_nueva)

    if es_repetida:
        print("❌ La imagen NO se guardó porque ya existe en el dataset.")
    else:
        X.append(imagen_nueva.flatten())
        y.append(clase_deseada)

        X_actualizado = np.array(X, dtype=np.float32)
        y_actualizado = np.array(y, dtype=np.int64)

        np.savez(ruta_archivo, X=X_actualizado, y=y_actualizado)
        print("✅ ¡Imagen ÚNICA! Se agregó y guardó en el archivo .npz.")
        print(f"Nuevo total de imágenes: {len(X_actualizado)}")


if __name__ == "__main__":
    ruta_dataset = os.path.join(os.path.dirname(__file__), "dataset_sintetico.npz")
    
    agregar_imagen_al_dataset(
        ruta_archivo=ruta_dataset, 
        clase_deseada=0, 
        flip_probability=0.08
    )
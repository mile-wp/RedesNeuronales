#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 11:34:31 2026

@author: walter
"""


#Archivo con los datos de entrenamiento
#Genera las imagenes

import numpy as np

# Definición del Dataset Sintético (Patrones 4x4 con ruido)
patterns = np.array([
    # Clase 0
    [[1,1,0,0],
     [1,1,0,0],
     [0,0,0,0],
     [0,0,0,0]],
    # Clase 1
    [[0,0,1,1],
     [0,0,1,1],
     [0,0,0,0],
     [0,0,0,0]],
    # Clase 2
    [[0,0,0,0],
     [0,0,0,0],
     [1,1,0,0],
     [1,1,0,0]],
    # Clase 3
    [[0,0,0,0],
     [0,0,0,0],
     [0,0,1,1],
     [0,0,1,1]]
], dtype=np.uint8)

def generate_dataset(n_per_class=500, flip_probability=0.08, seed=1):
    rng = np.random.default_rng(seed)
    X, y = [], []

    for label, pattern in enumerate(patterns):
        for _ in range(n_per_class):
            img = pattern.copy()
            noise = rng.random((4,4)) < flip_probability
            img = np.logical_xor(img, noise).astype(np.uint8)
            X.append(img.flatten())
            y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def limitar_imagenes_repetidas(X, y, max_repetidas=2, seed=42):
    rng = np.random.default_rng(seed)
    unicas, indices_inversos, conteos = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    
    indices_a_mantener = []
    for i, count in enumerate(conteos):
        idx_actuales = np.where(indices_inversos == i)[0]
        if count > max_repetidas:
            idx_seleccionados = rng.choice(idx_actuales, size=max_repetidas, replace=False)
            indices_a_mantener.extend(idx_seleccionados)
        else:
            indices_a_mantener.extend(idx_actuales)
            
    rng.shuffle(indices_a_mantener)
    
    print(f"\n[CONTROL DE DUPLICADOS] Muestras antes del filtro: {len(X)}")
    print(f"[CONTROL DE DUPLICADOS] Muestras después de limitar a máx. {max_repetidas} repeticiones: {len(indices_a_mantener)}")
    
    return X[indices_a_mantener], y[indices_a_mantener]

if __name__ == "__main__":
    print("Generando dataset...")
    X, y = generate_dataset(n_per_class=500, flip_probability=0.08, seed=1)
    
    print("Limitando imágenes repetidas...")
    X, y = limitar_imagenes_repetidas(X, y, max_repetidas=2, seed=42)
    
    # Guardar los datos en un archivo binario comprimido
    np.savez("dataset_sintetico.npz", X=X, y=y)
    print("\n[INFO] Dataset guardado exitosamente como 'dataset_sintetico.npz'")
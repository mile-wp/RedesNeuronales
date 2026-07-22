# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:21:43 2026

@author: walte
"""

import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Patrones base (4 clases)
# ------------------------------

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


# ------------------------------
# Generador
# ------------------------------

def generate_dataset(n_per_class=1000,
                     flip_probability=0.10,
                     seed=0):

    rng = np.random.default_rng(seed)

    X = []
    y = []

    for label, pattern in enumerate(patterns):

        for _ in range(n_per_class):

            img = pattern.copy()

            # invertir algunos bits
            noise = rng.random((4,4)) < flip_probability
            img = np.logical_xor(img, noise).astype(np.uint8)

            X.append(img.flatten())
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    return X, y


# ------------------------------
# Crear dataset
# ------------------------------

X, y = generate_dataset(
    n_per_class=500,
    flip_probability=0.08,
    seed=1
)

print(X.shape)
print(y.shape)

# ------------------------------
# Mostrar ejemplos
# ------------------------------

fig, ax = plt.subplots(2,8, figsize=(10,3))

for i, a in enumerate(ax.flat):
    idx = np.random.randint(len(X))
    a.imshow(X[idx].reshape(4,4), cmap='gray', vmin=0, vmax=1)
    a.set_title(y[idx])
    a.axis("off")

plt.tight_layout()
plt.show()
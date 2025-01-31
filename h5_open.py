import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File('trading_model_v3.h5', 'r') as f:
    # 1. Explorar la estructura (MUY importante)
    print("Contenido del archivo:")
    for key in f.keys():
        print(key)
        if isinstance(f[key], h5py.Group):
            print("  (Grupo):")
            grupo = f[key]  # Obtén el objeto grupo
            for subkey in grupo.keys():
                print("   ", subkey)
                if isinstance(grupo[subkey], h5py.Dataset):
                    print("      (Dataset)")
                    dataset = grupo[subkey]  # Accede al dataset DENTRO del grupo
                    print("      Shape:", dataset.shape)  # Ahora sí puedes acceder a shape
                    print("      Type:", dataset.dtype)  # Y también al tipo
                    try:
                        imagen = dataset[:]
                        if imagen.ndim == 2:
                            plt.imshow(imagen, cmap='gray')
                        elif imagen.ndim == 3:
                            plt.imshow(imagen)
                        else:
                            print("Dimensiones no soportadas para imshow")
                        plt.title(subkey) # Usar el nombre del dataset como título
                        plt.show()
                    except Exception as e:
                        print(f"Error con el dataset '{subkey}': {e}")

        elif isinstance(f[key], h5py.Dataset): #Manejo de datasets en la raíz
            dataset = f[key]
            print("   (Dataset)")
            print("      Shape:", dataset.shape)
            print("      Type:", dataset.dtype)
            try:
                imagen = dataset[:]
                if imagen.ndim == 2:
                    plt.imshow(imagen, cmap='gray')
                elif imagen.ndim == 3:
                    plt.imshow(imagen)
                else:
                    print("Dimensiones no soportadas para imshow")
                plt.title(key) # Usar el nombre del dataset como título
                plt.show()
            except Exception as e:
                print(f"Error con el dataset '{key}': {e}")
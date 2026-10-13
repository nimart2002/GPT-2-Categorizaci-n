"""Configuración del proyecto: todos los valores por defecto en un solo sitio.

Cada experimento crea estos objetos y cambia solo lo que necesite, por ejemplo:
    cfg = WorldConfig(eta=(0.2,) * 5)
"""
from dataclasses import dataclass


@dataclass
class WorldConfig:
    """Parámetros del mundo de categorías (sección 4 del documento del TFM)."""

    # Probabilidad de que cada propiedad F1..F5 se desvíe de su valor "de libro".
    # Son desviaciones persistentes del objeto: se sortean una vez y no cambian.
    eta: tuple = (0.1, 0.1, 0.1, 0.1, 0.1)

    # Probabilidad de "despiste" en cada resultado (Y1, Y2) en cada encuentro.
    # SUPUESTO: el documento no da su valor; usamos el mismo que eta. Confirmar con el tutor.
    epsilon: tuple = (0.1, 0.1)

    # Probabilidad de que el subtipo oculto z valga 1, y de que la propiedad irrelevante F6 valga 1.
    p_z: float = 0.5
    p_x6: float = 0.5

    # Nombres de las 4 categorías. Qué nombre corresponde a qué (u, v) se sortea en cada mundo.
    category_names: tuple = ("KAV", "MIP", "RUD", "ZET")

    

@dataclass
class DataConfig:
    """Cómo se convierten objetos y encuentros en ejemplos de entrenamiento."""

    # Separación por objeto: los objetos de prueba no aparecen nunca en el entrenamiento.
    n_train_objects: int = 1000
    n_test_objects: int = 500

    # Probabilidad de ocultar ("?") cada propiedad F1..F6 en un ejemplo de entrenamiento.
    # SUPUESTO: el documento pide "máscaras variadas" pero no da el valor.
    p_hide_feature: float = 0.3

    # Probabilidad de escribir "CAT ?" en vez de la categoría.
    # El documento da siempre la etiqueta en la fase 1, así que empezamos con 0.
    p_hide_category: float = 0.0

    # Semilla del mundo: fija los nombres de las categorías y los objetos.
    world_seed: int = 0


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 128       # longitud máxima de contexto
    n_layer: int = 2            # número de bloques
    n_head: int = 2             # cabezas por bloque
    n_embd: int = 64            # dimensión del flujo residual
    ff_dim: int = 256           # dimensión interna del MLP
    dropout: float = 0.0        # probabilidad de dropout (0 = desactivado)
    bias: bool = True           # sesgos en las capas lineales y LayerNorm


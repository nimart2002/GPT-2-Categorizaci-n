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
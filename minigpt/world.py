"""El mundo de categorías: genera objetos y encuentros (sección 4 del documento del TFM).

Variables ocultas (el modelo nunca las ve):
    u, v  -> definen la categoría c = (u, v)
    z     -> subtipo dentro de la categoría
    n1..n5 -> desviaciones persistentes de cada objeto
Propiedades visibles (F1..F6):
    x1 = u ⊕ n1        x2 = v ⊕ n2
    x3 = z ⊕ n3        x4 = u ⊕ z ⊕ n4        x5 = v ⊕ z ⊕ n5
    x6 ~ Bernoulli(1/2)   (irrelevante)
Encuentro con la acción a ∈ {0, 1}:
    y1 = u ⊕ z ⊕ a ⊕ ν1        y2 = v ⊕ y1 ⊕ ν2
(⊕ es XOR, la suma módulo 2; en Python se escribe ^)
"""
import random
from dataclasses import dataclass

from minigpt.config import WorldConfig


def bernoulli(p, rng):
    """1 con probabilidad p, 0 con probabilidad 1 - p."""
    return 1 if rng.random() < p else 0


@dataclass(frozen=True)
class Object:
    object_id: int
    category: str          # nombre visible, p. ej. "MIP"
    u: int                 # variables ocultas
    v: int
    z: int
    n: tuple               # (n1, ..., n5)
    x: tuple               # propiedades visibles (x1, ..., x6)


@dataclass(frozen=True)
class Encounter:
    object_id: int
    action: int            # a
    y: tuple               # resultados (y1, y2)
    slips: tuple           # (ν1, ν2), ocultos


class World:
    """Un mundo concreto: la asignación de nombres a categorías depende de la semilla."""

    def __init__(self, cfg: WorldConfig = None, seed: int = 0):
        self.cfg = cfg if cfg is not None else WorldConfig()
        self.rng = random.Random(seed)
        # Sortear qué nombre corresponde a cada categoría (u, v)
        names = list(self.cfg.category_names)
        self.rng.shuffle(names)
        self.name_of = {(0, 0): names[0], (0, 1): names[1], (1, 0): names[2], (1, 1): names[3]}
        self.uv_of = {name: uv for uv, name in self.name_of.items()}
        self._next_id = 0

    def sample_object(self):
        """Un objeto nuevo, con sus variables ocultas y sus propiedades visibles."""
        rng, cfg = self.rng, self.cfg
        u, v = rng.randint(0, 1), rng.randint(0, 1)         # p(c) = 1/4
        z = bernoulli(cfg.p_z, rng)
        n = tuple(bernoulli(eta, rng) for eta in cfg.eta)
        x = (
            u ^ n[0],
            v ^ n[1],
            z ^ n[2],
            u ^ z ^ n[3],
            v ^ z ^ n[4],
            bernoulli(cfg.p_x6, rng),
        )
        obj = Object(self._next_id, self.name_of[(u, v)], u, v, z, n, x)
        self._next_id += 1
        return obj

    def sample_encounter(self, obj: Object, action=None):
        """Un encuentro con obj. Si no se da la acción, se sortea."""
        rng = self.rng if rng is None else rng
        cfg = self.cfg
        a = rng.randint(0, 1) if action is None else action
        nu1 = bernoulli(cfg.epsilon[0], rng)
        nu2 = bernoulli(cfg.epsilon[1], rng)
        y1 = obj.u ^ obj.z ^ a ^ nu1
        y2 = obj.v ^ y1 ^ nu2
        return Encounter(obj.object_id, a, (y1, y2), (nu1, nu2))



if __name__ == "__main__":
    world = World(seed=0)
    print("Nombres de las categorías en este mundo:")
    for uv, name in world.name_of.items():
        print(f"  (u, v) = {uv}  ->  {name}")

    print("\nCinco objetos con un encuentro cada uno:")
    for _ in range(5):
        obj = world.sample_object()
        enc = world.sample_encounter(obj)
        props = " ".join(f"F{i + 1}={xi}" for i, xi in enumerate(obj.x))
        print(f"  objeto {obj.object_id}: {obj.category}  {props}  "
              f"acción={enc.action}  Y1={enc.y[0]} Y2={enc.y[1]}   "
              f"[ocultas: u={obj.u} v={obj.v} z={obj.z} n={obj.n}]")

    
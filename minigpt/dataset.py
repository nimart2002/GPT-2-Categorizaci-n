"""Del mundo a los tensores: separación por objeto, máscaras de observación y lotes.

Cada ejemplo es UN registro precedido de <bos> (22 tokens):
    <bos> CAT MIP F1 0 F2 ? ... ACT 0 Y1 1 Y2 0 END
Como todos miden lo mismo, no hace falta relleno (<pad>) en esta fase.
"""
import random

import torch

from minigpt.config import DataConfig
from minigpt.records import BOS, format_record, target_mask
from minigpt.world import World

IGNORE = -100     # valor que F.cross_entropy(ignore_index=-100) no tiene en cuenta


def split_objects(world: World, cfg: DataConfig):
    """Genera los objetos de entrenamiento y de prueba. Nunca comparten objeto."""
    train = [world.sample_object() for _ in range(cfg.n_train_objects)]
    test = [world.sample_object() for _ in range(cfg.n_test_objects)]
    return train, test


def sample_observed(cfg: DataConfig, rng):
    """Máscara de observación aleatoria: True = se ve, False = se escribe "?"."""
    return tuple(rng.random() >= cfg.p_hide_feature for _ in range(6))


def encode_example(tokenizer, obj, enc, observed=None, show_category=True):
    """Un registro -> (x, y), listas de enteros de la misma longitud.

    x: la entrada (todo menos el último token).
    y: el token siguiente en cada posición, o IGNORE si no es un objetivo.
    """
    tokens = [BOS] + format_record(obj, enc, observed, show_category)
    ids = tokenizer.encode(tokens)
    mask = target_mask(tokens)
    x = ids[:-1]
    y = [i if m else IGNORE for i, m in zip(ids[1:], mask[1:])]
    return x, y


def get_batch(world, objects, tokenizer, cfg: DataConfig, batch_size, rng):
    """Un lote: objetos al azar, cada uno con un encuentro NUEVO y una máscara nueva."""
    xs, ys = [], []
    for _ in range(batch_size):
        obj = rng.choice(objects)
        enc = world.sample_encounter(obj, rng=rng)
        observed = sample_observed(cfg, rng)
        show_category = rng.random() >= cfg.p_hide_category
        x, y = encode_example(tokenizer, obj, enc, observed, show_category)
        xs.append(x)
        ys.append(y)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long)


if __name__ == "__main__":
    from minigpt.records import build_record_tokenizer

    cfg = DataConfig()
    world = World(seed=cfg.world_seed)
    tok = build_record_tokenizer(world.cfg)
    train_objs, test_objs = split_objects(world, cfg)
    print(f"{len(train_objs)} objetos de entrenamiento, {len(test_objs)} de prueba")

    rng = random.Random(0)
    x, y = get_batch(world, train_objs, tok, cfg, batch_size=4, rng=rng)
    print("forma de x:", tuple(x.shape), "  forma de y:", tuple(y.shape))

    print("\nPrimer ejemplo del lote, posición a posición:")
    print(f"  {'pos':>3}  {'entrada':>7}  ->  objetivo")
    for t in range(x.shape[1]):
        inp = tok.itos[x[0, t].item()]
        tgt = y[0, t].item()
        tgt_str = tok.itos[tgt] if tgt != IGNORE else "(ignorado)"
        print(f"  {t:>3}  {inp:>7}  ->  {tgt_str}")
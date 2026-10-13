"""Registros: convierten un objeto y un encuentro en una secuencia de tokens.

Formato 21 tokens por registro:
    CAT MIP F1 0 F2 1 F3 1 F4 0 F5 0 F6 1 ACT 0 Y1 1 Y2 0 END
Lo que no se observa se escribe "?":
    - una propiedad oculta:      F4 ?
    - sin etiqueta de categoría: CAT ?
"""
from minigpt.config import WorldConfig
from minigpt.tokenizer import Tokenizer
from minigpt.world import Encounter, Object

PAD = "<pad>"        # relleno, para igualar longitudes en un lote
BOS = "<bos>"        # comienzo de secuencia
UNK = "?"            # valor no observado
END = "END"          # fin de registro

FEATURES = ["F1", "F2", "F3", "F4", "F5", "F6"]
OUTCOMES = ["Y1", "Y2"]
VALUES = ["0", "1"]

# Los campos cuyo VALOR (el token siguiente) es un objetivo de la pérdida.
TARGET_FIELDS = set(FEATURES + OUTCOMES)


def build_vocab(cfg: WorldConfig = None):
    cfg = cfg if cfg is not None else WorldConfig()
    return ([PAD, BOS, "CAT"] + list(cfg.category_names) + FEATURES
            + ["ACT"] + OUTCOMES + VALUES + [UNK, END])


def build_record_tokenizer(cfg: WorldConfig = None):
    return Tokenizer(build_vocab(cfg))


def format_record(obj: Object, enc: Encounter, observed=None, show_category=True):
    """Devuelve la lista de tokens de un registro.

    observed: tupla de 6 booleanos; observed[i] = False oculta Fi (se escribe "?").
              None significa que se observan todas.
    show_category: si es False, se escribe "CAT ?".
    """
    if observed is None:
        observed = (True,) * 6
    assert enc.object_id == obj.object_id, "el encuentro no es de este objeto"

    tokens = ["CAT", obj.category if show_category else UNK]
    for name, value, seen in zip(FEATURES, obj.x, observed):
        tokens += [name, str(value) if seen else UNK]
    tokens += ["ACT", str(enc.action)]
    tokens += ["Y1", str(enc.y[0]), "Y2", str(enc.y[1])]
    tokens.append(END)
    return tokens


def target_mask(tokens):
    """Para cada posición, True si ese token es un objetivo de la pérdida.

    Un token es objetivo si es el VALOR de una propiedad o de un resultado
    (el token anterior está en TARGET_FIELDS) y no es "?".
    """
    mask = [False] * len(tokens)
    for t in range(1, len(tokens)):
        if tokens[t - 1] in TARGET_FIELDS and tokens[t] != UNK:
            mask[t] = True
    return mask


if __name__ == "__main__":
    from minigpt.world import World

    tok = build_record_tokenizer()
    print(f"Vocabulario ({tok.vocab_size} tokens):")
    print(" ", tok.vocab)

    world = World(seed=0)
    obj = world.sample_object()
    enc = world.sample_encounter(obj)

    tokens = format_record(obj, enc)
    print("\nRegistro completo:")
    print(" ", " ".join(tokens))
    print("  números:", tok.encode(tokens))

    partial = format_record(obj, enc, observed=(True, True, True, False, True, False),
                            show_category=False)
    print("\nRegistro parcial (sin categoría, sin F4 ni F6):")
    print(" ", " ".join(partial))

    print("\n¿Qué tokens cuentan en la pérdida? (marcados con ^)")
    mask = target_mask(partial)
    width = [len(t) for t in partial]
    print(" ", " ".join(partial))
    print(" ", " ".join(("^" if m else " ").ljust(w) for m, w in zip(mask, width)))
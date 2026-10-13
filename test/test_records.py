"""Tests del formato de los registros y de la máscara de objetivos."""
from minigpt.records import (UNK, build_record_tokenizer, format_record,
                             target_mask)
from minigpt.world import World


def example():
    world = World(seed=0)
    obj = world.sample_object()
    return obj, world.sample_encounter(obj)


def test_vocab_is_small_and_unique():
    tok = build_record_tokenizer()
    assert tok.vocab_size == 20
    assert len(set(tok.vocab)) == tok.vocab_size


def test_record_length_is_21():
    obj, enc = example()
    assert len(format_record(obj, enc)) == 21


def test_record_contains_the_right_values():
    obj, enc = example()
    tokens = format_record(obj, enc)
    assert tokens[:2] == ["CAT", obj.category]
    for i in range(6):
        assert tokens[2 + 2 * i] == f"F{i + 1}"
        assert tokens[3 + 2 * i] == str(obj.x[i])
    assert tokens[14:] == ["ACT", str(enc.action), "Y1", str(enc.y[0]),
                           "Y2", str(enc.y[1]), "END"]


def test_round_trip_through_tokenizer():
    obj, enc = example()
    tok = build_record_tokenizer()
    tokens = format_record(obj, enc)
    assert tok.decode(tok.encode(tokens)) == tokens


def test_hidden_values_are_not_in_the_record():
    obj, enc = example()
    tokens = format_record(obj, enc, observed=(True, True, True, False, True, True),
                           show_category=False)
    assert tokens[1] == UNK
    assert tokens[8:10] == ["F4", UNK]          # F4 está, pero su valor no
    assert len(tokens) == 21                    # la longitud no cambia


def test_target_mask_full_record():
    obj, enc = example()
    tokens = format_record(obj, enc)
    mask = target_mask(tokens)
    targets = [i for i, m in enumerate(mask) if m]
    # valores de F1..F6 (posiciones 3,5,...,13) y de Y1, Y2 (17, 19)
    assert targets == [3, 5, 7, 9, 11, 13, 17, 19]


def test_target_mask_skips_unknowns_category_and_action():
    obj, enc = example()
    tokens = format_record(obj, enc, observed=(True, True, True, False, True, True))
    mask = target_mask(tokens)
    assert not mask[1]           # la categoría condiciona, pero no es objetivo
    assert not mask[9]           # F4 = "?" no es objetivo
    assert not mask[15]          # la acción la elige el agente: no se predice
    assert sum(mask) == 7
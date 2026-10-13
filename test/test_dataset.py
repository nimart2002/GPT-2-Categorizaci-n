"""Tests de la separación por objeto, las máscaras y los lotes."""
import random

import pytest

from minigpt.config import DataConfig
from minigpt.dataset import IGNORE, encode_example, get_batch, split_objects
from minigpt.records import UNK, build_record_tokenizer
from minigpt.world import World


@pytest.fixture
def setup():
    cfg = DataConfig(n_train_objects=200, n_test_objects=100)
    world = World(seed=cfg.world_seed)
    tok = build_record_tokenizer(world.cfg)
    train, test = split_objects(world, cfg)
    return cfg, world, tok, train, test


def test_split_by_object(setup):
    cfg, world, tok, train, test = setup
    train_ids = {o.object_id for o in train}
    test_ids = {o.object_id for o in test}
    assert len(train_ids) == 200 and len(test_ids) == 100
    assert train_ids.isdisjoint(test_ids)


def test_batch_shapes(setup):
    cfg, world, tok, train, test = setup
    x, y = get_batch(world, train, tok, cfg, batch_size=8, rng=random.Random(0))
    assert x.shape == (8, 21) and y.shape == (8, 21)


def test_y_is_x_shifted_where_not_ignored(setup):
    cfg, world, tok, train, test = setup
    x, y = get_batch(world, train, tok, cfg, batch_size=8, rng=random.Random(0))
    keep = y[:, :-1] != IGNORE
    assert (y[:, :-1][keep] == x[:, 1:][keep]).all()


def test_full_record_has_8_targets(setup):
    cfg, world, tok, train, test = setup
    obj = train[0]
    enc = world.sample_encounter(obj)
    x, y = encode_example(tok, obj, enc)
    assert sum(1 for t in y if t != IGNORE) == 8


def test_hidden_feature_is_neither_input_nor_target(setup):
    cfg, world, tok, train, test = setup
    obj = train[0]
    enc = world.sample_encounter(obj)
    observed = (True, True, True, False, True, True)        # ocultamos F4
    x, y = encode_example(tok, obj, enc, observed)
    tokens = tok.decode(x)
    i = tokens.index("F4")
    assert tokens[i + 1] == UNK          # en la entrada solo hay "?"
    assert y[i] == IGNORE                # y no se le pide al modelo que lo prediga


def test_hide_rate_matches_config(setup):
    cfg, world, tok, train, test = setup
    x, y = get_batch(world, train, tok, cfg, batch_size=2000, rng=random.Random(1))
    unk = tok.stoi[UNK]
    rate = (x == unk).float().sum().item() / (2000 * 6)
    assert abs(rate - cfg.p_hide_feature) < 0.02


def test_batches_never_use_test_objects(setup):
    cfg, world, tok, train, test = setup
    rng = random.Random(2)
    seen = {rng.choice(train).object_id for _ in range(1000)}
    assert seen <= {o.object_id for o in train}
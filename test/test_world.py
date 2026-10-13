"""Comprobamos que el mundo cumple las probabilidades del documento del TFM."""
import pytest

from minigpt.config import WorldConfig
from minigpt.world import World

N = 20_000          # muchos objetos para que las frecuencias se acerquen a las probabilidades
TOL = 0.015         # margen permitido


@pytest.fixture(scope="module")
def objects():
    world = World(seed=123)
    return [world.sample_object() for _ in range(N)]


def freq(condition, items):
    items = list(items)
    return sum(1 for it in items if condition(it)) / len(items)


def test_names_are_a_bijection():
    world = World(seed=0)
    assert sorted(world.name_of.values()) == sorted(WorldConfig().category_names)
    for uv, name in world.name_of.items():
        assert world.uv_of[name] == uv


def test_categories_are_uniform(objects):
    for uv in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert abs(freq(lambda o: (o.u, o.v) == uv, objects) - 0.25) < TOL


def test_f1_reveals_u_with_prob_0_9(objects):
    assert abs(freq(lambda o: o.x[0] == o.u, objects) - 0.9) < TOL


def test_relational_feature_0_82(objects):
    # El número clave: P(x4 = x3 | u = 0) = (1-η)² + η² = 0.82
    u0 = [o for o in objects if o.u == 0]
    assert abs(freq(lambda o: o.x[3] == o.x[2], u0) - 0.82) < TOL


def test_f6_is_independent_noise(objects):
    assert abs(freq(lambda o: o.x[5] == 1, objects) - 0.5) < TOL
    assert abs(freq(lambda o: o.x[5] == o.u, objects) - 0.5) < TOL


def test_deviations_are_fixed_per_object():
    world = World(seed=1)
    obj = world.sample_object()
    xs = {obj.x for _ in range(10)}        # el objeto no cambia entre encuentros
    assert len(xs) == 1


def test_outcome_y1_rule():
    world = World(seed=2)
    obj = world.sample_object()
    encs = [world.sample_encounter(obj, action=1) for _ in range(N)]
    rule = obj.u ^ obj.z ^ 1
    assert abs(freq(lambda e: e.y[0] == rule, encs) - 0.9) < TOL


def test_config_override():
    cfg = WorldConfig(eta=(0.0,) * 5, epsilon=(0.0, 0.0))
    world = World(cfg, seed=3)
    for _ in range(200):
        o = world.sample_object()
        assert o.x[:5] == (o.u, o.v, o.z, o.u ^ o.z, o.v ^ o.z)
        e = world.sample_encounter(o)
        assert e.y[0] == o.u ^ o.z ^ e.action
        assert e.y[1] == o.v ^ e.y[0]
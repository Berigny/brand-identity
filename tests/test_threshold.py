import math

def threshold(prime, base=0.2, k=0.6):
    return base + k * (1 - prime)

def test_peripheral_easy():
    assert math.isclose(threshold(1.0), 0.2)

def test_core_hard():
    assert math.isclose(threshold(0.0), 0.8)

def test_midline():
    assert math.isclose(threshold(0.5), 0.5)

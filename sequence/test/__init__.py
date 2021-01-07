import pytest
import random
from itertools import permutations
from sequence.data.utils import Language, Dataset


@pytest.fixture(scope="module")
def words():
    return list("ABCDEFGHIJJKLMNOPQRST")


@pytest.fixture(scope="module")
def paths(words):
    # Create random sequences with random length
    random.seed(1)
    perm = permutations("".join(words))
    return [next(perm)[: random.choice(range(1, 9))] for _ in range(2000)]


@pytest.fixture(scope="module")
def paths_long(words):
    # Create random sequences with random length
    random.seed(1)
    perm = permutations("".join(words))
    return [next(perm)[: random.choice(range(1, 9))] for _ in range(68783)]


@pytest.fixture(scope="module")
def language(words):
    return Language(words)


@pytest.fixture(scope="module")
def dataset(paths, language):
    return Dataset(paths, language)

from sequence.data.utils import DatasetInference
from sequence.model.stamp import STAMP
from sequence.test import language, paths, paths_long, words
import torch


def test_batch_inference(language, paths):
    torch.manual_seed(0)
    m = STAMP(language.vocabulary_size, embedding_dim=2)

    dataset = DatasetInference(
        sentences=paths, language=language, min_len=4, max_len=10
    )

    with torch.no_grad():
        batch_size = 50
        packed_padded, padded = dataset.get_batch(0, batch_size)
        out = m(packed_padded)
        # Dim 1, number of batches
        assert out.shape[0] == batch_size
        # Dim 2, the number of elements in longest sequence
        assert out.shape[1] == padded.abs().argmin(0).max() + 1
        # Dim 3, vocabulary_size
        assert out.shape[2] == dataset.language.vocabulary_size


def test_inference_ds(language, paths, paths_long):
    torch.manual_seed(0)

    dataset = DatasetInference(
        sentences=paths, language=language, min_len=1, max_len=100
    )
    assert len(dataset) == len(paths), "dataset short failed"

    dataset_long = DatasetInference(
        sentences=paths_long, language=language, min_len=1, max_len=100, mask=True
    )
    assert len(dataset_long) == len(paths_long), "dataset long failed"

    dataset_long = DatasetInference(
        sentences=paths_long, language=language, min_len=1, max_len=100, mask=False
    )
    assert len(dataset_long) != len(paths_long), "dataset long failed"

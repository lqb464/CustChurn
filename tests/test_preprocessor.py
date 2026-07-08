import numpy as np

from src.data.preprocessor import split_train_validation_test
from src.data.schema import TARGET_COLUMN


def test_split_is_reproducible_and_70_15_15(sample_frame):
    first = split_train_validation_test(sample_frame, TARGET_COLUMN)
    second = split_train_validation_test(sample_frame, TARGET_COLUMN)
    assert [len(part) for part in first] == [84, 18, 18]
    assert np.array_equal(first[0][TARGET_COLUMN], second[0][TARGET_COLUMN])

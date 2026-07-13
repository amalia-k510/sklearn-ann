import numpy as np
import pytest

from sklearn_ann.test_utils import assert_row_close, needs

try:
    from hannoy import Metric

    from sklearn_ann.kneighbors.hannoy import HannoyTransformer
except ImportError:
    pass


@needs.hannoy
def test_euclidean(random_small, random_small_pdists):
    trans = HannoyTransformer(metric=Metric.EUCLIDEAN)
    mat = trans.fit_transform(random_small)
    euclidean_dist = random_small_pdists["euclidean"]
    assert_row_close(mat, euclidean_dist)


@needs.hannoy
def test_cosine(random_small, random_small_pdists):
    trans = HannoyTransformer(metric=Metric.COSINE)
    mat = trans.fit_transform(random_small)
    # hannoy's cosine metric returns (1 - cos_sim) / 2
    cosine_dist = random_small_pdists["cosine"] / 2
    assert_row_close(mat, cosine_dist)


@needs.hannoy
def test_transform_matches_fit_transform(random_small):
    # fit_transform uses the by_item path; fit().transform() uses by_vec
    # comparing the two paths
    trans = HannoyTransformer(metric=Metric.EUCLIDEAN)
    by_item = trans.fit_transform(random_small)
    by_vec = trans.fit(random_small).transform(random_small)
    by_item.sort_indices()
    by_vec.sort_indices()
    np.testing.assert_array_equal(by_item.indices, by_vec.indices)
    np.testing.assert_allclose(by_item.data, by_vec.data, atol=1e-5)

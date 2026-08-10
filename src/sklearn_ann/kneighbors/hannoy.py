# hannoy needs a filesystem path (LMDB-backed)
from __future__ import annotations

import tempfile
from itertools import count
from typing import TYPE_CHECKING, cast

import numpy as np
from hannoy import Database, Metric
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags, TargetTags, TransformerTags
from sklearn.utils.validation import validate_data

from ..utils import TransformerChecksMixin

if TYPE_CHECKING:
    from typing import Self

    from numpy.typing import NDArray

# guards against panic in Rust, i.e. unrecoverable errors
SUPPORTED_M = frozenset({4, 8, 12, 16, 24, 32})

# due to hannoy sharing one global LMDB environement
# using to avoid colliding
_index_counter = count()


class HannoyTransformer(TransformerChecksMixin, TransformerMixin, BaseEstimator):
    """Wrap :class:`hannoy.Database` as a scikit-learn ``KNeighborsTransformer``.

    Notes
    -----
    Known issue is that hannoy only creates one LMDB env per processor.
    Once HannoyTransfor sets the path; every other case of transformer
    will reuse this same env.

    It is harmless as it is being handled in a way that each transformer
    gets its own index, so no overwriting is taking place. For real isolation,
    run them in separate processes.
    """

    n_neighbors: int
    """Number of neighbors to return."""

    metric: Metric | None
    """Distance metric. ``None`` means euclidean."""

    path: str | None
    """LMDB directory. ``None`` creates a temp dir."""

    m: int
    """Edges per node in the HNSW graph. One of {4, 8, 12, 16, 24, 32}."""

    ef_construction: int
    """Candidate list size when building (higher = better graph, slower)."""

    ef_search: int
    """Candidate list size when searching (higher = better recall, slower)."""

    def __init__(
        self,
        n_neighbors: int = 5,
        *,
        metric: Metric | None = None,
        path: str | None = None,
        m: int = 16,
        ef_construction: int = 96,
        ef_search: int = 200,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.metric = metric
        # LMDB directory for the index; if None = auto-create a temp dir
        self.path = path
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search

    def fit(self, X: NDArray[np.number], y: None = None) -> Self:
        X = cast(
            "NDArray[np.float32]", validate_data(self, X, dtype=np.float32, order="C")
        )
        # guard to avoid panic abort
        if self.m not in SUPPORTED_M:
            raise ValueError(
                f"m={self.m!r} is not supported by hannoy; "
                f"choose one of {sorted(SUPPORTED_M)}."
            )
        self.n_samples_fit_ = X.shape[0]
        metric = Metric.EUCLIDEAN if self.metric is None else self.metric
        path = (
            self.path if self.path is not None else tempfile.mkdtemp(prefix="hannoy_")
        )
        self._index_ = next(_index_counter) % 2**16

        # metric is fixed for the entire database
        self.hannoy_db_ = Database(path, metric)
        with self.hannoy_db_.writer(
            X.shape[1], index=self._index_, m=self.m, ef=self.ef_construction
        ) as writer:
            writer.add_items(list(range(self.n_samples_fit_)), X)
        self.hannoy_reader_ = self.hannoy_db_.reader(index=self._index_)
        return self

    def transform(self, X: NDArray[np.number]) -> csr_matrix[np.float32]:
        # verify that fit was called and + that X has the right number of features
        X = cast(
            "NDArray[np.float32]",
            self._transform_checks(X, "hannoy_reader_", dtype=np.float32, order="C"),
        )
        return self._transform(X)

    def fit_transform(  # type: ignore[override]
        self, X: NDArray[np.number], y: None = None
    ) -> csr_matrix[np.float32]:
        self.fit(X)
        X = cast(
            "NDArray[np.float32]",
            validate_data(self, X, dtype=np.float32, order="C", reset=False),
        )
        return self._transform(X)

    def _transform(self, X: NDArray[np.float32]) -> csr_matrix[np.float32]:
        # how many points
        n_samples_transform = X.shape[0]
        n_neighbors = self.n_neighbors + 1
        # pre allocating indicies for which points are neighbots
        indices = np.empty((n_samples_transform, n_neighbors), dtype=np.uint32)
        distances = np.empty((n_samples_transform, n_neighbors), dtype=np.float32)
        self.hannoy_reader_.by_array(
            X, n=n_neighbors, ef_search=self.ef_search, out=(indices, distances)
        )

        metric = Metric.EUCLIDEAN if self.metric is None else self.metric
        if metric == Metric.EUCLIDEAN:
            # hannoy's EUCLIDEAN returns squared distance; sqrt to get the true distance
            np.sqrt(distances, out=distances)

        indptr = np.arange(0, n_samples_transform * n_neighbors + 1, n_neighbors)
        return csr_matrix(
            (distances.ravel(), indices.ravel(), indptr),
            shape=(n_samples_transform, self.n_samples_fit_),
        )

    def __sklearn_tags__(self) -> Tags:
        # metadata
        return Tags(
            estimator_type="transformer",
            target_tags=TargetTags(required=False),
            transformer_tags=TransformerTags(preserves_dtype=["float32"]),
        )

"""Article-aware no-duplicates batch sampler.

``BatchSamplers.NO_DUPLICATES`` only forbids equal *texts* in a batch. With
in-batch negatives that is not enough here: two questions about the same article
usually have different positive chunks, so each would treat a chunk of its own
gold article as a negative. This sampler forbids two rows touching the same
article (positive or negative side) in one batch; equal texts imply equal
articles, so it is strictly stronger than NO_DUPLICATES.
"""

from collections.abc import Iterator, Sequence

import torch
from sentence_transformers.base.sampler import DefaultBatchSampler


class ArticleNoDuplicatesBatchSampler(DefaultBatchSampler):
    def __init__(self, dataset, batch_size: int, drop_last: bool, row_keys: Sequence[frozenset[str]], **kwargs) -> None:
        super().__init__(dataset, batch_size=batch_size, drop_last=drop_last, **kwargs)
        if len(row_keys) != len(dataset):
            raise ValueError("row_keys must be aligned with the dataset")
        self.row_keys = row_keys
        self.epoch = 0

    def __iter__(self) -> Iterator[list[int]]:
        if self.generator is not None and self.seed is not None:
            self.generator.manual_seed(self.seed + self.epoch)
        remaining = torch.randperm(len(self.row_keys), generator=self.generator).tolist()
        while remaining:
            batch: list[int] = []
            used: set[str] = set()
            deferred: list[int] = []
            for pos, idx in enumerate(remaining):
                keys = self.row_keys[idx]
                if used.isdisjoint(keys):
                    batch.append(idx)
                    used |= keys
                    if len(batch) == self.batch_size:
                        deferred.extend(remaining[pos + 1 :])
                        break
                else:
                    deferred.append(idx)
            remaining = deferred
            if len(batch) < self.batch_size and self.drop_last:
                return
            yield batch

    def __len__(self) -> int:
        n = len(self.row_keys)
        return n // self.batch_size if self.drop_last else -(-n // self.batch_size)


def make_sampler_factory(row_keys: Sequence[frozenset[str]]):
    def factory(dataset, batch_size, drop_last, valid_label_columns=None, generator=None, seed=0):
        return ArticleNoDuplicatesBatchSampler(
            dataset,
            batch_size,
            drop_last,
            row_keys=row_keys,
            valid_label_columns=valid_label_columns,
            generator=generator,
            seed=seed,
        )

    return factory

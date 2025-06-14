from torch.utils.data import Dataset, IterableDataset
from random import choices

from ..misc.step_tracker import StepTracker
from .dataset_re10k import DatasetRE10k, DatasetRE10kCfg
from .types import Stage
from .view_sampler import get_view_sampler, ViewSamplerCfg, VIEW_SAMPLER_CFGS



DatasetCfg = DatasetRE10kCfg


class MixedDataset(IterableDataset):
    def __init__(self, datasets: list[Dataset], weights: list[float]) -> None:
        super().__init__()
        self.datasets = datasets
        self.weights = weights
        assert len(self.datasets) == len(self.weights), "Number of datasets must match number of weights."
        self.num_dataset = len(datasets)
        self.idxs = range(self.num_dataset)
        self.remain_counts = [0] * self.num_dataset
        self.dataset_iterators = [None] * self.num_dataset

    def __iter__(self):
        for _ in range(len(self)):
            # Choose a dataset based on the weights.
            idx = choices(self.idxs, weights=self.weights, k=1)[0]
            if self.remain_counts[idx] == 0:
                self.dataset_iterators[idx] = iter(self.datasets[idx])
                self.remain_counts[idx] = len(self.datasets[idx])
            self.remain_counts[idx] -= 1
            yield next(self.dataset_iterators[idx])

    def __len__(self):
        return sum(len(dataset) for dataset in self.datasets)


def get_mixed_dataset(
    cfgs: dict,
    stage: Stage,
    step_tracker: StepTracker | None,
) -> Dataset:
    datasets, weights = [], []
    for name, cfg in cfgs.items():
        if type(cfg) != DatasetCfg: continue # view_sampler is config group
        view_sampler_dict = cfgs["view_sampler"][cfg.view_sampler]
        view_sampler = get_view_sampler(
            VIEW_SAMPLER_CFGS[cfg.view_sampler](**view_sampler_dict),
            stage,
            cfg.overfit_to_scene is not None,
            cfg.cameras_are_circular,
            step_tracker,
        )
        dataset = DatasetRE10k(cfg, stage, view_sampler)
        datasets.append(dataset)
        weights.append(cfg.weight)

    return MixedDataset(datasets, weights)


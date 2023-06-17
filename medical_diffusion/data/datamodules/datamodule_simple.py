import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
import torch.multiprocessing as mp
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler


class SimpleDataModule(pl.LightningDataModule):

    def __init__(self,
                 ds_train: object,
                 ds_val: object = None,
                 ds_test: object = None,
                 batch_size: int = 1,
                 num_workers: int = mp.cpu_count(),
                 seed: int = 0,
                 pin_memory: bool = False,
                 weights: list = None
                 ):
        super().__init__()
        self.hyperparameters = {**locals()}
        self.hyperparameters.pop('__class__')
        self.hyperparameters.pop('self')

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test

        self.batch_size = batch_size
        self.num_workers = num_workers  # 用于数据加载的并行工作线程数（默认为CPU核心数量）
        self.seed = seed
        self.pin_memory = pin_memory  # 是否将数据保存在pin memory区，pin memory中的数据转到GPU会快一些
        self.weights = weights  # 样本权重是一种用于处理类别不平衡问题的技术，它给予某些样本更高的权重

    def train_dataloader(self):
        # 这是使用pytorch框架自带的随机数生成器 torch.Generator()，用于生成随机数
        generator = torch.Generator()
        '''在每个 epoch 中，RandomSampler 会对数据集中的样本进行随机打乱，并以随机顺序进行加载。
        种子（seed）的作用是在每次重新运行时产生相同的随机顺序，但在同一次运行中，不同的 epoch 之间仍然是随机的。'''
        generator.manual_seed(self.seed)

        if self.weights is not None:
            sampler = WeightedRandomSampler(self.weights, len(self.weights), generator=generator)
        else:
            # RandomSampler 返回的是整个数据集的随机索引
            # RandomSampler 返回一个迭代器对象，该迭代器对象用于生成随机抽样的索引。
            # 当你使用 RandomSampler 类创建一个实例后，可以通过对该实例进行迭代来获取随机抽样的索引。
            # 迭代器对象将按照类中定义的逻辑生成索引 这取决于抽样方式（有放回或无放回）以及指定的抽样数量。
            sampler = RandomSampler(self.ds_train, replacement=False, generator=generator)

        # DataLoader 在每个 epoch 中只采样与抽样器（sampler）对应的随机索引的数据。
        # 这意味着 DataLoader 将根据sampler返回的索引来访问数据集，并将仅获取与这些索引对应的样本数据。
        # DataLoader 并不会一次性采样完整的数据集，而是根据抽样器返回的索引逐个获取样本数据，每个索引对应一个样本。
        return DataLoader(self.ds_train, batch_size=self.batch_size, num_workers=self.num_workers,
                          sampler=sampler, generator=generator, drop_last=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_val is not None:
            # 设置种子和设置 shuffle=False 的作用是不同的。种子主要影响每次运行时的随机性
            # shuffle=False 在每个 epoch 中，数据加载器将以数据集中定义的顺序被加载,而不进行随机打乱
            return DataLoader(self.ds_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                              generator=generator, drop_last=False, pin_memory=self.pin_memory)
        else:
            raise AssertionError("A validation set was not initialized.")

    def test_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        if self.ds_test is not None:
            return DataLoader(self.ds_test, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False,
                              generator=generator, drop_last=False, pin_memory=self.pin_memory)
        else:
            raise AssertionError("A test test set was not initialized.")

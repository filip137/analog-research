from abc import ABC, abstractmethod
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import random_split, DataLoader, TensorDataset



class IndexedDataset(torch.utils.data.Dataset):
    """
    Wrapper class that turns a (labeled) datadet into an `indexed dataset'.

    Each item of an `indexed dataset' consists of an image, its label, and its index in the dataset.
    In contrast, the standard Dataset class does not return the index of the image

    Attributes
    ----------
    _dataset (Dataset): the original dataset that we wrap into an `indexed dataset'
    """

    def __init__(self, dataset):
        """Creates an instance of IndexedDataset

        Args:
            dataset (Dataset): the original dataset that we wrap into an `indexed dataset'
        """
        self._dataset = dataset
        
    def __getitem__(self, index):
        """Returns the image, its label, and its index."""
        data, target = self._dataset[index]
        
        return data, target, index

    def __len__(self):
        return len(self._dataset)


class Datasets(ABC):
    def __init__(self, name, batch_size, device):
        self.name = name
        self.batch_size = batch_size
        self.device = device

    @abstractmethod
    def build(self):
        pass

class MoonsDataset(Datasets):
    def __init__(self, name, batch_size, device, num_samples, input_dim=None):
        super().__init__(name, batch_size, device)
        self.num_samples = num_samples
        self.noise = 0.1
        self.input_dim = int(input_dim) if input_dim is not None else None

    @staticmethod
    def _expand_xy_features(x, target_dim):
        if x.shape[1] == target_dim:
            return x
        if x.shape[1] != 2:
            raise ValueError(f"Expected 2D inputs for moons expansion, got shape {tuple(x.shape)}.")
        if target_dim < 2 or target_dim % 2 != 0:
            raise ValueError(f"Moons input_dim must be an even number >= 2; got {target_dim}.")
        half = target_dim // 2
        x1 = x[:, [0]].repeat(1, half)
        x2 = x[:, [1]].repeat(1, half)
        return torch.cat([x1, x2], dim=1)
    def build(self):
        from sklearn.datasets import make_moons

        x, y = make_moons(n_samples=self.num_samples, noise=self.noise, random_state=42)
        # Keep features in float32 to match model weights; labels stay int for targets.
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        # Normalize each feature to [-1, 1] based on global min/max.
        eps = 1e-12
        x_min = x.min(dim=0).values
        x_max = x.max(dim=0).values
        scale = (x_max - x_min).clamp_min(eps)
        x = (x - x_min) / scale * 2.0 - 1.0
        if self.input_dim is not None and self.input_dim != x.shape[1]:
            x = self._expand_xy_features(x, self.input_dim)
        dataset = TensorDataset(x, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_ds, test_ds = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(0))
        test_ds = IndexedDataset(test_ds)
        train_data_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        test_data_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        return train_data_loader, test_data_loader


class YinYangDataset(Datasets):
    """
    Balanced 2D Yin-Yang classification dataset with 3 classes:
    yin (0), yang (1), and dots (2).
    """

    def __init__(
        self,
        name,
        batch_size,
        device,
        num_samples,
        *,
        r_small=0.1,
        r_big=0.5,
        seed=42,
    ):
        super().__init__(name, batch_size, device)
        self.num_samples = int(num_samples)
        self.r_small = float(r_small)
        self.r_big = float(r_big)
        self.seed = int(seed)

    def _dist_to_right_dot(self, x, y):
        return np.sqrt((x - 1.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def _dist_to_left_dot(self, x, y):
        return np.sqrt((x - 0.5 * self.r_big) ** 2 + (y - self.r_big) ** 2)

    def _which_class(self, x, y):
        d_right = self._dist_to_right_dot(x, y)
        d_left = self._dist_to_left_dot(x, y)

        criterion1 = d_right <= self.r_small
        criterion2 = d_left > self.r_small and d_left <= 0.5 * self.r_big
        criterion3 = y > self.r_big and d_right > 0.5 * self.r_big
        is_yin = criterion1 or criterion2 or criterion3

        is_dot = d_right < self.r_small or d_left < self.r_small
        if is_dot:
            return 2
        return int(is_yin)

    def _sample_xy_for_class(self, rng, goal_class):
        while True:
            x, y = rng.rand(2) * 2.0 * self.r_big
            if np.sqrt((x - self.r_big) ** 2 + (y - self.r_big) ** 2) > self.r_big:
                continue
            c = self._which_class(x, y)
            if c == goal_class:
                return x, y

    def build(self):
        rng = np.random.RandomState(self.seed)
        xs = []
        ys = []

        # Keep classes balanced by target-class rejection sampling.
        for _ in range(self.num_samples):
            goal_class = int(rng.randint(3))
            x, y = self._sample_xy_for_class(rng, goal_class)
            xs.append([x, y])
            ys.append(goal_class)

        x = torch.tensor(np.asarray(xs), dtype=torch.float32, device=self.device)
        y = torch.tensor(np.asarray(ys), dtype=torch.long, device=self.device)

        # Match moons preprocessing range.
        x = x * 2.0 - 1.0

        dataset = TensorDataset(x, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_ds, test_ds = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(0),
        )
        test_ds = IndexedDataset(test_ds)
        train_data_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        test_data_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        return train_data_loader, test_data_loader


class LinSpaceDataset(Datasets):
    def __init__(self, name, batch_size, device, num_inputs, min, max, num_samples=100):
        super().__init__(name, batch_size, device)
        self.min = min
        self.max = max
        self.num_inputs = num_inputs
        self.num_samples = num_samples

    def build(self):
        grid = torch.linspace(self.min, self.max, self.num_samples, device=self.device)
        x = torch.zeros((self.num_samples, self.num_inputs), device=self.device)
        x[:, 0] = grid
        if self.num_inputs > 1:
            x[:, 1] = torch.sin(grid)
        labels = torch.zeros(self.num_samples, dtype=torch.long, device=self.device)
        dataset = TensorDataset(x, labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

    def build_mesh(self, xmin, xmax, ymin, ymax):
        x = torch.linspace(xmin, xmax, self.num_samples, device = self.device)
        y = torch.linspace(ymin, ymax, self.num_samples, device = self.device)
        X, Y = torch.meshgrid(x, y, indexing="ij")
        points = torch.stack([X.flatten(), Y.flatten()], dim=1)  # (N*N, 2)
        labels = torch.zeros(points.shape[0], dtype=torch.long, device=self.device)  # dummy labels if needed
        dataset = TensorDataset(points)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class DigitsDataset(Datasets):
    """Sklearn digits (8x8) dataset flattened to 64 features."""

    def __init__(self, name, batch_size, device, num_samples=None, seed=0):
        super().__init__(name, batch_size, device)
        self.num_samples = int(num_samples) if num_samples is not None else None
        self.seed = int(seed)

    def build(self):
        from sklearn.datasets import load_digits

        digits = load_digits()
        x = digits.data  # (N, 64), values in [0, 16]
        y = digits.target
        if self.num_samples is not None and self.num_samples < x.shape[0]:
            rng = np.random.RandomState(self.seed)
            idx = rng.permutation(x.shape[0])[: self.num_samples]
            x = x[idx]
            y = y[idx]
        x = torch.tensor(x, dtype=torch.float32, device=self.device)
        y = torch.tensor(y, dtype=torch.long, device=self.device)
        # Normalize to [-1, 1] (digits are in [0, 16]).
        x = x / 16.0 * 2.0 - 1.0

        dataset = TensorDataset(x, y)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_ds, test_ds = random_split(
            dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(self.seed),
        )
        test_ds = IndexedDataset(test_ds)
        train_data_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=False)
        test_data_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False)
        return train_data_loader, test_data_loader
def _build_torchvision_image_loaders(
    *,
    dataset_cls,
    batch_size,
    root,
    download,
    normalize,
    normalize_mean,
    normalize_std,
    normalize_scale=1.0,
):
    transforms_list = [transforms.ToTensor()]
    if normalize:
        transforms_list.append(
            transforms.Normalize(
                mean=(float(normalize_mean),), std=(float(normalize_std),)
            )
        )
        normalize_scale = float(normalize_scale)
        if normalize_scale != 1.0:
            transforms_list.append(
                transforms.Lambda(
                    lambda tensor, scale=normalize_scale: tensor * scale
                )
            )
    transform = transforms.Compose(transforms_list)

    train_dataset = dataset_cls(
        root=root,
        train=True,
        download=download,
        transform=transform,
    )
    test_dataset = dataset_cls(
        root=root,
        train=False,
        download=download,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class MnistDataset(Datasets):

    def __init__(
        self,
        name,
        batch_size,
        device,
        root,
        train,
        download,
        normalize,
        normalize_std,
        normalize_mean=0.1307,
        normalize_scale=1.0,
    ):
        super().__init__(name, batch_size, device)
        self.root = root
        self.train = train
        self.download = download
        self.normalize = normalize
        self.normalize_std = normalize_std
        self.normalize_mean = normalize_mean
        self.normalize_scale = normalize_scale

    def build(self):
        return _build_torchvision_image_loaders(
            dataset_cls=datasets.MNIST,
            batch_size=self.batch_size,
            root=self.root,
            download=self.download,
            normalize=self.normalize,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            normalize_scale=self.normalize_scale,
        )


class FashionMnistDataset(Datasets):

    def __init__(
        self,
        name,
        batch_size,
        device,
        root,
        train,
        download,
        normalize,
        normalize_std=0.3530,
        normalize_mean=0.2860,
        normalize_scale=1.0,
    ):
        super().__init__(name, batch_size, device)
        self.root = root
        self.train = train
        self.download = download
        self.normalize = normalize
        self.normalize_std = normalize_std
        self.normalize_mean = normalize_mean
        self.normalize_scale = normalize_scale

    def build(self):
        return _build_torchvision_image_loaders(
            dataset_cls=datasets.FashionMNIST,
            batch_size=self.batch_size,
            root=self.root,
            download=self.download,
            normalize=self.normalize,
            normalize_mean=self.normalize_mean,
            normalize_std=self.normalize_std,
            normalize_scale=self.normalize_scale,
        )


class TinyGridDataset(Datasets):
    """Synthetic dataset of tiny 2D grids for quick conv debugging."""

    def __init__(
        self,
        name,
        batch_size,
        device,
        patterns,
        samples_per_pattern=32,
        noise_std=0.0,
    ):
        super().__init__(name, batch_size, device)
        if not patterns:
            raise ValueError("TinyGridDataset requires at least one pattern.")
        self.patterns = patterns
        self.samples_per_pattern = samples_per_pattern
        self.noise_std = noise_std

    def build(self):
        xs = []
        ys = []

        for label, pattern in enumerate(self.patterns):
            base = torch.tensor(pattern, dtype=torch.float32)
            if base.ndim == 2:
                base = base.unsqueeze(0)  # add channel dimension
            elif base.ndim != 3:
                raise ValueError(f"Pattern must be 2D or 3D (got shape {tuple(base.shape)})")
            base = base.unsqueeze(0)  # add batch dimension → (1, C, H, W)

            for _ in range(self.samples_per_pattern):
                sample = base.clone()
                if self.noise_std > 0.0:
                    sample += self.noise_std * torch.randn_like(sample)
                xs.append(sample)
                ys.append(label)

        x_tensor = torch.cat(xs, dim=0).to(self.device)
        y_tensor = torch.tensor(ys, dtype=torch.long, device=self.device)
        dataset = TensorDataset(x_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

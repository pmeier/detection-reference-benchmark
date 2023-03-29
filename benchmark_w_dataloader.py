import sys

from utils import (
    benchmark,
    dataset_v1,
    dataset_v2,
    transform_v1,
    transform_v2,
    InMemoryDataset,
)

from torch.utils.data import DataLoader


def main(root):
    print("#" * 60)
    print("# Benchmark with DataLoader")
    print("#" * 60)
    for label, dataset_builder, transform_builder in [
        ("v1", dataset_v1, transform_v1),
        ("v2", dataset_v2, transform_v2),
    ]:
        print(label)
        print("-" * 60)

        print("Building dataset")
        dataset = dataset_builder(root)

        print("Caching dataset")
        dataset = InMemoryDataset(dataset, num_samples=5_000)

        transform = transform_builder()

        data_loader = DataLoader(
            dataset,
            num_workers=4,
            batch_size=24,
            collate_fn=list,
        )

        print("Benchmarking")
        medians = benchmark(data_loader, transform)
        for label, median in medians.items():
            print(f"{label}: {median * 1e6:.2f} µs/img")

        print("-" * 60)


if __name__ == "__main__":
    main(sys.argv[1])

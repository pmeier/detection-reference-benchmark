import sys

from utils import (
    benchmark,
    dataset_v1,
    dataset_v2,
    transform_v1,
    transform_v2,
    InMemoryDataset,
)


def main(root):
    print("#" * 60)
    print("# Benchmark without DataLoader")
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
        dataset = InMemoryDataset(dataset, num_samples=1_000)

        transform = transform_builder()

        print("Benchmarking")
        medians = benchmark(dataset, transform)
        for label, median in medians.items():
            print(f"{label}: {median * 1e6:.2f}µs")

        print("-" * 60)


if __name__ == "__main__":
    main(sys.argv[1])

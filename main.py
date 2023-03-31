import contextlib
import itertools
import pathlib
import sys
from datetime import datetime

import torch
import torchvision
from torch.utils.collect_env import main as collect_env

torchvision.disable_beta_transforms_warning()

from tasks import make_task


class Tee:
    def __init__(self, stdout, root=pathlib.Path(__file__).parent / "results"):
        self.stdout = stdout
        self.root = root
        self.file = open(root / f"{datetime.utcnow():%Y%m%d%H%M%S}.log", "w")

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)


def main(*, input_types, tasks, num_samples):
    # This is hardcoded when using a DataLoader with multiple workers:
    # https://github.com/pytorch/pytorch/blob/19162083f8831be87be01bb84f186310cad1d348/torch/utils/data/_utils/worker.py#L222
    torch.set_num_threads(1)

    for task_name in tasks:
        print("#" * 60)
        print(task_name)
        print("#" * 60)

        medians = {input_type: {} for input_type in input_types}
        for input_type in input_types:
            dataset_rng = torch.Generator()
            dataset_rng.manual_seed(0)
            dataset_rng_state = dataset_rng.get_state()

            for api_version in ["v1", "v2"]:
                dataset_rng.set_state(dataset_rng_state)
                task = make_task(
                    task_name,
                    input_type=input_type,
                    api_version=api_version,
                    dataset_rng=dataset_rng,
                    num_samples=num_samples,
                )
                if task is None:
                    continue

                print(f"{input_type=}, {api_version=}")
                print()
                print(f"Results computed for {num_samples:_} samples")
                print()

                pipeline, dataset = task

                for sample in dataset:
                    pipeline(sample)

                results = pipeline.extract_times()
                field_len = max(len(name) for name in results)
                print(f"{' ' * field_len}  {'median   ':>9}    {'std   ':>9}")
                medians[input_type][api_version] = 0.0
                for transform_name, times in results.items():
                    median = float(times.median())
                    print(
                        f"{transform_name:{field_len}}  {median * 1e6:6.0f} µs +- {float(times.std()) * 1e6:6.0f} µs"
                    )
                    medians[input_type][api_version] += median

                print(
                    f"\n{'total':{field_len}}  {medians[input_type][api_version] * 1e6:6.0f} µs"
                )
                print("-" * 60)

        print()
        print("Summaries")
        print()

        field_len = max(len(input_type) for input_type in medians)
        print(f"{' ' * field_len}  v2 / v1")
        for input_type, api_versions in medians.items():
            if len(api_versions) < 2:
                continue

            print(
                f"{input_type:{field_len}}  {api_versions['v2'] / api_versions['v1']:>7.2f}"
            )

        print()

        median_ref = medians["PIL"]["v1"]
        medians_flat = {
            f"{input_type}, {api_version}": median
            for input_type, api_versions in medians.items()
            for api_version, median in api_versions.items()
        }
        field_len = max(len(label) for label in medians_flat)
        print(f"{' ' * field_len}  x / PIL, v1")
        for label, median in medians_flat.items():
            print(f"{label:{field_len}}  {median / median_ref:>11.2f}")


if __name__ == "__main__":
    tee = Tee(stdout=sys.stdout)

    with contextlib.redirect_stdout(tee):
        main(
            tasks=["classification-simple", "classification-complex"],
            input_types=["Tensor", "PIL", "Datapoint"],
            num_samples=10_000,
        )

        print("#" * 60)
        collect_env()

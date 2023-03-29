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
    for task_name, input_type in itertools.product(tasks, input_types):
        print("#" * 60)
        print(f"{task_name=}, {input_type=}")
        print("#" * 60)

        dataset_rng = torch.Generator()
        dataset_rng.manual_seed(0)
        dataset_rng_state = dataset_rng.get_state()
        total_time = {}
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

            print(f"{api_version=}\n")

            pipeline, dataset = task

            for sample in dataset:
                pipeline(sample)

            results = pipeline.extract_times()
            field_len = max(len(name) for name in results)
            print(f"{' ' * field_len}  {'median   ':>9}    {'std   ':>9}")
            total_time[api_version] = 0.0
            for transform_name, times in results.items():
                median = float(times.median())
                print(
                    f"{transform_name:{field_len}}  {median * 1e6:6.0f} µs +- {float(times.std()) * 1e6:6.0f} µs"
                )
                total_time[api_version] += median

            print(f"\n{'total':{field_len}}  {total_time[api_version] * 1e6:6.0f} µs")
            print("-" * 60)

        if len(total_time) < 2:
            continue

        abs_slowdown = total_time["v2"] - total_time["v1"]
        rel_slowdown = abs_slowdown / total_time["v1"]
        print(
            f"v2 is {abs(abs_slowdown) * 1e6:.0f} µs/img ({abs(rel_slowdown):.1%}) "
            f"{'slower' if abs_slowdown > 0 else '*faster*'} than v1"
        )


if __name__ == "__main__":
    tee = Tee(stdout=sys.stdout)

    with contextlib.redirect_stdout(tee):
        collect_env()
        main(
            tasks=["classification-simple"],
            input_types=["Tensor", "PIL", "Datapoint"],
            num_samples=10_000,
        )

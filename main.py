import sys
import contextlib
import pathlib
from datetime import datetime

from torch.utils.collect_env import main as collect_env

from benchmark_wo_dataloader import main as benchmark_wo_dataloader
from benchmark_w_dataloader import main as benchmark_w_dataloader


class Tee:
    def __init__(self, stdout, root=pathlib.Path(__file__).parent / "results"):
        self.stdout = stdout
        self.root = root
        self.file = open(root / f"{datetime.utcnow():%Y%m%d%H%M%S}.log", "w")

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)


if __name__ == "__main__":
    root = sys.argv[1]

    tee = Tee(stdout=sys.stdout)

    with contextlib.redirect_stdout(tee):
        collect_env()
        benchmark_wo_dataloader(root)
        benchmark_w_dataloader(root)

import sys
from pcmr.data import data


def center(word, fill: str = '=', width: int = 30):
    return f"{word:{fill}^{width}}"


def list_all():
    with (open(sys.argv[1], "w") if len(sys.argv) >= 2 else sys.stdout) as f:
        for d in data.datasets:
            ts = data.get_tasks(d)
            if len(ts) > 1:
                [print(f"{d}/{t}", file=f) for t in ts]
            else:
                print(d, file=f)

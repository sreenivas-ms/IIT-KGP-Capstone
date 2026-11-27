"""Build the P6 smoke question/gold set from the repository fixtures (not the frozen test set)."""
from verisheet.experiments.smoke import build_smoke_set

if __name__ == "__main__":
    for kind, path in build_smoke_set().items():
        print(f"{kind}: {path}")

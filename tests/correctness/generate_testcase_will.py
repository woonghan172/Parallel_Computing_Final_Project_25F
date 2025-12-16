#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple N-body integer test inputs.")
    parser.add_argument("size", type=int, help="Number of bodies (rows)")
    parser.add_argument("index", type=int, help="Output file index used in filenames")
    parser.add_argument("--outdir", type=Path, default=Path("."), help="Output directory (default: current)")
    parser.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility (default: none)")
    parser.add_argument("--coord-min", type=int, default=-1000, help="Min coordinate integer (default: -1000)")
    parser.add_argument("--coord-max", type=int, default=1000, help="Max coordinate integer (default: 1000)")
    parser.add_argument("--mass-min", type=int, default=1, help="Min mass integer (default: 1)")
    parser.add_argument("--mass-max", type=int, default=1000, help="Max mass integer (default: 1000)")
    args = parser.parse_args()

    if args.size <= 0:
        raise SystemExit("size must be > 0")
    if args.coord_max < args.coord_min:
        raise SystemExit("coord-max must be >= coord-min")
    if args.mass_max < args.mass_min:
        raise SystemExit("mass-max must be >= mass-min")

    rng = random.Random(args.seed)

    args.outdir.mkdir(parents=True, exist_ok=True)

    coord_path = args.outdir / f"testin{args.index}_coordinate.txt"
    mass_path = args.outdir / f"testin{args.index}_mass.txt"

    with coord_path.open("w", encoding="utf-8") as fc, mass_path.open("w", encoding="utf-8") as fm:
        for _ in range(args.size):
            x = rng.randint(args.coord_min, args.coord_max)
            y = rng.randint(args.coord_min, args.coord_max)
            z = rng.randint(args.coord_min, args.coord_max)
            m = rng.randint(args.mass_min, args.mass_max)

            fc.write(f"{x} {y} {z}\n")
            fm.write(f"{m}\n")

if __name__ == "__main__":
    main()

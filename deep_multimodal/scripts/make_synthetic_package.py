"""Write a small synthetic model input package (random values, real layout) for tests and notebooks. Usage: python scripts/make_synthetic_package.py <folder> [year]"""
import sys

from deep_multimodal.synthetic import write_synthetic_package

if __name__ == "__main__":
    folder = write_synthetic_package(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 2022)
    print("wrote synthetic package under", folder)

"""Convenient workspace entry point; the implementation lives with the operators."""
import argparse
from operators.build import build


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', action='store_true', help='Optimize for this CPU')
    build(parser.parse_args().native)

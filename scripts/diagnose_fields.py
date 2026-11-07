"""Compatibility entry point for the compact-checkpoint-aware field audit."""
import argparse
from audit_predicted_fields import run


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('config', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output)

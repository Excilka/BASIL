import argparse
import json

from cmsir.config import load_config
from cmsir.engine import test as run_test
from cmsir.utils import configure_threads, require_cuda

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_threads(cfg["runtime"]["threads"])
    device = require_cuda(cfg["device"], cfg["runtime"]["minimum_free_memory_mib"])
    print(json.dumps(run_test(cfg, device, args.checkpoint), indent=2))

if __name__ == "__main__":
    main()

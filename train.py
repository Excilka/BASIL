import argparse
import json

from cmsir.config import load_config
from cmsir.engine import test, train
from cmsir.utils import configure_threads, require_cuda, set_determinism

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--test-after-train", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    configure_threads(cfg["runtime"]["threads"])
    device = require_cuda(cfg["device"], cfg["runtime"]["minimum_free_memory_mib"])
    set_determinism(cfg["seed"])
    print(json.dumps(train(cfg, device), indent=2), flush=True)
    if args.test_after_train:
        print(json.dumps(test(cfg, device), indent=2), flush=True)

if __name__ == "__main__":
    main()

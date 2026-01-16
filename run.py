from __future__ import annotations
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import json
import yaml
import os

from config import cfg, contextcfg, predcfg
from datagen.generate_data import main as datagen_main
from train.train import main as train_main
from eval.eval import main as eval_main




def make_run_dir(run_name: str, root: str = "runs") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(root) / f"{ts}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir

def snapshot_configs(run_dir: Path, cfg: cfg, contextcfg: contextcfg, predcfg: predcfg, extra: dict):
    (run_dir / "meta").mkdir(exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump({
            "Config": asdict(cfg),
            "EGNNconfig": asdict(contextcfg),
            "PredictorConfig": asdict(predcfg),
        }, f, sort_keys=False)

    with open(run_dir / "meta" / "run.json", "w") as f:
        json.dump(extra, f, indent=2)

def main():

    run_name = "jepamd"
    out_dir = make_run_dir(run_name)

    cfg_ = cfg()
    ctxcfg = contextcfg()
    pcfg = predcfg()

    snapshot_configs(
        out_dir, cfg_, ctxcfg, pcfg,
        extra={"run_name": run_name}
    )

    datagen_main(cfg_, cfg_.seed_train, out_dir=out_dir, is_train=True)

    datagen_main(cfg_, cfg_.seed_val, out_dir=out_dir, is_train=False)
    
    os.mkdir(f'{out_dir}/checkpoints')

    train_main(cfg_, ctxcfg, pcfg, seed=cfg_.seed_train, out_dir=out_dir)

    load_epoch = cfg_.load_epoch

    ckpt_path = f'{out_dir}/checkpoints/epoch_{load_epoch}.eqx'

    eval_main(cfg_, ctxcfg, pcfg, cfg_.seed_train, out_dir, ckpt_dir=ckpt_path)
    
    print(f"Done. Run saved to: {out_dir}")

if __name__ == "__main__":
    main()

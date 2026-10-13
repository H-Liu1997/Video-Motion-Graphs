import os
import random
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from utils.config import parse_args
from vmgv10_trainer import CustomTrainer


@logger.catch(reraise=True)
def main():
    args = parse_args()
    Path(args.out_path).mkdir(parents=True, exist_ok=True)
    logger.add(str(Path(args.out_path) / "run.log"))
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    trainer = CustomTrainer(args)
    if not args.preprocess_only:
        trainer.train(0)


if __name__ == "__main__":
    main()

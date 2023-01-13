import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import time
import logging
import psutil
import torch
import tqdm
from fvcore.nn import flop_count_table  # can also try flop_count_str
from torch.nn.parallel import DistributedDataParallel

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.data import (
    DatasetFromList,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.benchmark import DataLoaderBenchmark
from detectron2.engine import AMPTrainer, SimpleTrainer, default_argument_parser, hooks, launch
from detectron2.modeling import build_model
from detectron2.utils import comm
from detectron2.utils.collect_env import collect_env_info
from detectron2.utils.events import CommonMetricPrinter
from detectron2.utils.logger import setup_logger
from detectron2.utils.analysis import (
    FlopCountAnalysis,
    activation_count_operators,
    parameter_count_table,
)

logger = logging.getLogger("detectron2")


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    setup_logger(distributed_rank=comm.get_rank())
    return cfg


if __name__ == "__main__":
    parser = default_argument_parser()
    args = parser.parse_args()
    assert not args.eval_only

    logger.info("Environment info:\n" + collect_env_info())
    cfg = setup(args)
    # input_shape = (3, 1152, 1152)
    input_shape = (1, 3, 1152, 1152)

    cfg.dataloader.num_workers = 0
    data_loader = instantiate(cfg.dataloader.test)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    # DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)
    model.eval()

    num_warmup = 50
    log_interval = 50
    pure_inf_time = 0
    total_iters = 200
    for idx, data in zip(tqdm.trange(total_iters), data_loader):
        # for idx in tqdm.trange(total_iters):
        # data[0]["image"] = torch.randn(input_shape)
        data = torch.randn(input_shape).cuda()

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        with torch.no_grad():
            # model(data)
            model.backbone(data)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if idx >= num_warmup:
            pure_inf_time += elapsed

    fps = (total_iters - num_warmup) / pure_inf_time
    logger.info("Overall fps: {:.2f} img / s".format(fps))
    logger.info("Times per image: {:2f} s".format(1 / fps))

    flops = FlopCountAnalysis(model.backbone, data)
    # flops = FlopCountAnalysis(model, data)
    logger.info(
        "Flops table computed from only one input sample:\n" + flop_count_table(flops)
    )
    logger.info(f"Flops: {flops.total() / 1e9} G")

import sys
PROJ_ROOT = "."
sys.path.insert(0, PROJ_ROOT)
import os
os.chdir("../..")
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

from core.gdrn_modeling.main_gdrn import setup

from core.utils.default_args_setup import my_default_argument_parser, my_default_setup

from detectron2.data import MetadataCatalog
from core.gdrn_modeling.data_loader import build_gdrn_train_loader, build_gdrn_test_loader

import numpy as np
import matplotlib.pyplot as plt
# args = my_default_argument_parser().parse_args(['--config-file', 'output/gdrn/lm/a6_cPnP_lm13_norm_14/a6_cPnP_lm13_norm.py', '--num-gpus', '1'])
# args = my_default_argument_parser().parse_args(['--config-file', 'configs/gdrn/ycbv/a6_cPnP_AugAAETrunc_BG0.5_Rsym_ycbv_real_pbr_visib20_10e_norm.py', '--num-gpus', '1'])
args = my_default_argument_parser().parse_args(['--config-file', '/home/tianyou_zhang/Project/NVR-Net/gdr-normal-NVR-Net/output/a6_cPnP_airplane_norm_23_debug/airplane.py', '--num-gpus', '1'])

cfg = setup(args)
# cfg.OUTPUT_DIR = 'debug/gdrn/ycbv/a6_cPnP_AugAAETrunc_BG0.5_Rsym_ycbv_real_pbr_visib20_10e_norm_test'
cfg.OUTPUT_DIR = 'debug/gdrn/airplane/a6_cPnP_AugAAETrunc_BG0.5_lmo_real_pbr0.1_40e_norm_efficientpose/'
cfg.TEST.USE_SVD = False

dataset_meta = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
train_dset_names = cfg.DATASETS.TRAIN[0]
print(train_dset_names)
data_loader = build_gdrn_test_loader(cfg, train_dset_names)
data_loader_iter = iter(data_loader)

data = next(data_loader_iter)

from core.gdrn_modeling.models import GDRN
model, optimizer = eval(cfg.MODEL.CDPN.NAME).build_model_optimizer(cfg)

from core.utils.my_checkpoint import MyCheckpointer
from torch.cuda.amp import GradScaler
import core.utils.my_comm as comm
cfg.MODEL.WEIGHTS = '/home/tianyou_zhang/Project/NVR-Net/gdr-normal-NVR-Net/output/a6_cPnP_airplane_norm_23_debug/model_final.pth'
# cfg.MODEL.WEIGHTS = '/home/fgkun/projects/GDR-Net/output/gdrn/lmo/a6_cPnP_AugAAETrunc_BG0.5_lmo_real_pbr0.1_40e_norm_1/model_final.pth'
# cfg.MODEL.WEIGHTS = '/home/fgkun/projects/GDR-Net/output/gdrn/lm/a6_cPnP_lm13_norm_4/model_0102399.pth'
from core.utils import solver_utils
scheduler = solver_utils.build_lr_scheduler(cfg, optimizer, total_iters=1)
grad_scaler = GradScaler()
checkpointer = MyCheckpointer(
        model,
        cfg.OUTPUT_DIR,
        optimizer=optimizer,
        scheduler=scheduler,
        gradscaler=grad_scaler,
        save_to_disk=comm.is_main_process(),
    )
start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True).get("iteration", -1) + 1

from core.gdrn_modeling.engine_utils import batch_data
from detectron2.utils.events import EventStorage
with EventStorage(start_iter) as storage:
    batch = batch_data(cfg, data)

from core.gdrn_modeling.engine_utils import batch_data
from detectron2.utils.events import EventStorage
with EventStorage(start_iter) as storage:
#     if np.random.rand() < train_2_ratio:
#         data = next(data_loader_2_iter)
#     else:
#         data = next(data_loader_iter)
    batch = batch_data(cfg, data)
    out_dict = model(
        batch["roi_img"],
        roi_classes=batch["roi_cls"],
        roi_cams=batch["roi_cam"],
        roi_whs=batch["roi_wh"],
        roi_centers=batch["roi_center"],
        resize_ratios=batch["resize_ratio"],
        roi_coord_2d=batch.get("roi_coord_2d", None),
        roi_extents=batch.get("roi_extent", None),
    )
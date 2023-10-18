import os
import os.path as osp
import sys
import time
import numpy as np
from tqdm import tqdm
import mmcv

cur_dir = osp.dirname(osp.abspath(__file__))
PROJ_ROOT = osp.normpath(osp.join(cur_dir, "../../.."))
sys.path.insert(0, PROJ_ROOT)

import ref
import ref.airplane as airplane
from lib.utils.utils import dprint, iprint, lazy_property
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
import cv2

class AirplaneDataset(object):

    def __init__(self, data_cfg):
        self.dataset_name = data_cfg["dataset_name"]
        self.height = data_cfg["height"]
        self.width = data_cfg["width"]
        self.data_file = data_cfg["data_file"]
        
        self.ann_file = data_cfg["ann_file"]
        self.image_prefixes = data_cfg["image_prefixes"]
        self.camera_file = data_cfg["camera_file"]
        # ann_files = [cur_ann_file]


    def __call__(self):
        dataset_dict = []
        camera_file = self.camera_file
        ann_file = self.ann_file
        image_prefixes = self.image_prefixes
        dirs = os.listdir(image_prefixes)
        indices = len(dirs)
        for dir in dirs:
            if osp.splitext(dir)[1] != ".png":
                indices -= 1
                continue

        with open(camera_file, "r") as f:
            _, width, height, focal, dx, x, y ,z = f.readline().strip("\r\n").split("\t")
            K = np.zeros((3, 3), dtype=np.float32)
            K[0,0] = int(focal) / float(dx)
            K[1,1] = int(focal) / float(dx)
            K[0,2] = int(width) / 2
            K[1,2] = int(height) / 2
            K[2,2] = 1
            f.close()
            
        with open(ann_file, "r") as f:
            ### id left_x: top_y: width: height:
            annos = []
            for line in f.readlines():
                id, _, left_x, _, top_y, _, width, _, height = line.strip("\r\n").split(" ")
                annos.append([float(left_x), float(top_y), float(width), float(height)])
            f.close()

        for im_id, bbox in enumerate(tqdm(annos)):
            image_path = osp.join(image_prefixes, "{:d}.png").format(im_id)
            assert osp.exists(image_path), image_path

            record = {
                "dataset_name": self.dataset_name,
                "file_name": osp.abspath(image_path),
                "height": self.height,
                "width": self.width,
                "cam": K,
                "scene_im_id": f"{int(0)}/{im_id + 1}",
                "annotations": {
                    "bbox": bbox,
                    "bbox_mode": BoxMode.XYWH_ABS,
                },
            }

            dataset_dict.append(record)
        return dataset_dict

def get_airplane_metadata(obj_names, ref_key):
    data_ref = ref.__dict__[ref_key]
    meta = {"thing_classes": obj_names}
    return meta

AIRPLANE_OBJECT = ["airplane"]

################ register datasets with follow config############################
airplane_data = dict(
    dataset_name = airplane.dataset_name,
    data_file = airplane.data_file, # /output/airplane/zuo3
    height = airplane.height,
    width = airplane.width,
    objs = AIRPLANE_OBJECT,
    ann_file = osp.join(airplane.data_file, "bbox.txt"),
    image_prefixes = airplane.data_file,
    camera_file=
        osp.join(airplane.data_file, "{}_focal.txt").format(airplane.data_name),
    ref_key = "airplane",
)

def register_with_name_cfg(name, data_cfg=None):
    """Assume pre-defined datasets live in `./datasets`.

    Args:
        name: datasnet_name,
        data_cfg: if name is in existing SPLITS, use pre-defined data_cfg
            otherwise requires data_cfg
            data_cfg can be set in cfg.DATA_CFG.name
    """
    dprint("register dataset: {}".format(name))
    used_cfg = airplane_data if data_cfg is None else data_cfg
    DatasetCatalog.register(name, AirplaneDataset(used_cfg))
    # something like eval_types
    MetadataCatalog.get(name).set(
        objs=used_cfg["objs"],
        id="test_airplane",  # NOTE: for pvnet to determine module
        ref_key=used_cfg["ref_key"],
        eval_error_types=["ad", "rete", "proj"],
        evaluator_type="bop",
        **get_airplane_metadata(obj_names=used_cfg["objs"], ref_key=used_cfg["ref_key"]),
    )

def get_available_datasets():
    return list(["airplane"])

#### tests ###############################################
def test_vis():
    dset_name = "airplane"
    assert dset_name in DatasetCatalog.list()

    meta = MetadataCatalog.get(dset_name)
    dprint("MetadataCatalog: ", meta)
    objs = meta.name

    t_start = time.perf_counter()
    dicts = DatasetCatalog.get(dset_name)
    logger.info("Done loading {} samples with {:.3f}s.".format(len(dicts), time.perf_counter() - t_start))

    dirname = "output/{}-data-vis".format(dset_name)
    os.makedirs(dirname, exist_ok=True)
    for d in dicts:
        img = read_image_cv2(d["image_name"], format="BGR")
        imH, imW = img.shape[:2]

        anno = d["annotations"]
        bbox = anno["bbox"]
        bbox_mode = anno["bbox_mode"]
        bboxes_xyxy = [BoxMode.convert(bbox, bbox_mode, BoxMode.XYXY_ABS)]
        # 0-based label
        # cat_ids = [anno["category_id"] for anno in annos]
        K = d["camera"]
        # # TODO: visualize pose and keypoints
        for _i in range(len(dicts)):
            box_image = cv2.rectangle(img.copy(), (int(bboxes_xyxy[_i][0]), int(bboxes_xyxy[_i][1])), (int(bboxes_xyxy[_i][2]), int(bboxes_xyxy[_i][3])), (0, 255, 0), 2)
            grid_show(
                [
                    img[:, :, [2, 1, 0]],
                    box_image[:, :, [2, 1, 0]],
                ],
                [
                    "img",
                    "box_img",
                ],
                row=1,
                col=2,
            )

if __name__ == "__main__":
    """Test the  dataset loader.

    python this_file.py dataset_name
    """
    from lib.vis_utils.image import grid_show
    from lib.utils.setup_logger import setup_my_logger

    import detectron2.data.datasets  # noqa # add pre-defined metadata
    from lib.vis_utils.image import vis_image_mask_bbox_cv2
    from core.utils.utils import get_emb_show
    from core.utils.data_utils import read_image_cv2

    print("sys.argv:", sys.argv)
    logger = setup_my_logger(name="core")
    register_with_name_cfg("airplane")
    print("dataset catalog: ", DatasetCatalog.list())

    test_vis()
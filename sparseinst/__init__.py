from .sparseinst import SparseInst
from .encoder import build_sparse_inst_encoder
from .decoder import build_sparse_inst_decoder
from .config import add_sparse_inst_config
from .loss import build_sparse_inst_criterion
from .dataset_mapper import SparseInstDatasetMapper
from .coco_evaluation import COCOMaskEvaluator
from .backbones import build_resnet_vd_backbone, build_pyramid_vision_transformer
from .d2_predictor import VisualizationDemo

from detectron2.data.datasets import register_coco_instances
train_data = "coco-550-550"
test_data = "coco-550-550-val"
register_coco_instances(train_data, {}, f"./datasets/{train_data}/annotations/segments.json", f"./datasets/{train_data}/images/")
register_coco_instances(test_data, {}, f"./datasets/{test_data}/annotations/segments.json", f"./datasets/{test_data}/images/")

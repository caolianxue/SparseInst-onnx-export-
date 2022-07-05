'''
将pth文件转换为onnx
'''
import argparse
from detectron2.config import get_cfg
from sparseinst import add_sparse_inst_config
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader
from detectron2.export import Caffe2Tracer
from sparseinst.caffe2sparseinst import Caffe2SparseInst
import onnx
from pathlib import Path
import onnxoptimizer

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.MODEL.DEVICE = 'cpu'
    cfg.freeze()
    return cfg

def get_parser():
    parser = argparse.ArgumentParser(
        description="Detectron2 demo for builtin models")
    parser.add_argument(
        "--config-file",
        default="configs/quick_schedules/mask_rcnn_R_50_FPN_inference_acc_test.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true",
                        help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.15,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

if __name__ == '__main__':
    args = get_parser().parse_args()

    pth_path = './output/sparse_inst_r50_giam/model_final.pth'

    args.config_file = 'configs/sparse_inst_r50_giam.yaml'
    args.output = 'results'
    args.opts = ['MODEL.WEIGHTS', pth_path, 'INPUT.MIN_SIZE_TEST', '550']

    cfg = setup_cfg(args)

    # get a sample data
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    first_batched = next(iter(data_loader))
    input = [first_batched[0]]

    model = build_model(cfg)
    DetectionCheckpointer(model).resume_or_load(cfg.MODEL.WEIGHTS)
    model.eval()

    onnx_path = pth_path.replace('pth', 'onnx')
    onnx_model =  Caffe2Tracer(cfg, model, input).export_onnx()
    onnx_model = onnxoptimizer.optimize(onnx_model)
    onnx.save(onnx_model, onnx_path)

    if Path(onnx_path).exists():
        model = onnx.load(onnx_path)
        if model.ir_version < 4:
            print("Model with ir_version below 4 requires to include initilizer in graph input")

        inputs = model.graph.input
        name_to_input = {}
        for input in inputs:
            name_to_input[input.name] = input

        for initializer in model.graph.initializer:
            if initializer.name in name_to_input:
                inputs.remove(name_to_input[initializer.name])

        onnx.save(model, onnx_path)
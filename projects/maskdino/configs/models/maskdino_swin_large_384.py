from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import SwinTransformer

from .maskdino_r50 import model


# modify backbone config
model.backbone = L(SwinTransformer)(
    pretrain_img_size=384,
    embed_dim=192,
    depths=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 48),
    window_size=12,
    out_indices=(0, 1, 2, 3),
)

# modify neck config
input_shape = {'p0': ShapeSpec(channels=192, height=None, width=None, stride=8), 'p1': ShapeSpec(channels=384, height=None, width=None, stride=16), 'p2': ShapeSpec(channels=768, height=None, width=None, stride=32), 'p3': ShapeSpec(channels=1536, height=None, width=None, stride=64)}
model.sem_seg_head.input_shape = input_shape
model.sem_seg_head.pixel_decoder.input_shape = input_shape
model.sem_seg_head.pixel_decoder.transformer_in_features = ["p0", "p1", "p2", "p3"]
model.sem_seg_head.pixel_decoder.total_num_feature_levels = 5
model.sem_seg_head.transformer_predictor.total_num_feature_levels = 5


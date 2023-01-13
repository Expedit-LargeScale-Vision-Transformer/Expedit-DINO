from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.backbone import SwinTransformer
from ...modeling.backbone.hourglass_swin import HourglassSwinTransformer

from .maskdino_swin_large_384 import model


# modify backbone config
model.backbone = L(HourglassSwinTransformer)(
    pretrain_img_size=384,
    embed_dim=192,
    depths=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 48),
    window_size=12,
    out_indices=(0, 1, 2, 3),
    token_clustering_cfg=dict(
        num_spixels=81,
        n_iters=3,
        temperture=1e2,
        window_size=5,
    ),
    clustering_location=10,
    token_reconstruction_cfg=dict(
        k=25,
        temperture=1e2,
    ),
)


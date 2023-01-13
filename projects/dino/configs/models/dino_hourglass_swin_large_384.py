from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from ...modeling.backbone.hourglass_swin import HourglassSwinTransformer

from .dino_r50 import model


# modify backbone config
model.backbone = L(HourglassSwinTransformer)(
    pretrain_img_size=384,
    embed_dim=192,
    depths=(2, 2, 18, 2),
    num_heads=(6, 12, 24, 48),
    window_size=12,
    out_indices=(1, 2, 3),
    token_clustering_cfg=dict(
        num_spixels=64,
        n_iters=3,
        temperture=1e2,
        window_size=5,
    ),
    clustering_location=14,
    token_reconstruction_cfg=dict(
        k=25,
        temperture=1e2,
    ),
)

# modify neck config
model.neck.input_shapes = {
    "p1": ShapeSpec(channels=384),
    "p2": ShapeSpec(channels=768),
    "p3": ShapeSpec(channels=1536),
}
model.neck.in_features = ["p1", "p2", "p3"]

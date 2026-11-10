"""One place to construct the velocity net from a model config.

Every call site (train, inference, vram_check) goes through build_flow_net so
they share the same forward signature.
"""

from src.models.flow_unet import FlowUNet


def build_flow_net(mcfg):
    out_ch = mcfg.get("out_ch", 1)
    in_ch = mcfg.get("in_ch")
    concat_source = mcfg.get("concat_source", True)
    num_fields = mcfg.get("num_fields", 0)

    return FlowUNet(
        out_ch=out_ch, in_ch=in_ch, concat_source=concat_source,
        base=mcfg.get("base", 64), mults=tuple(mcfg.get("mults", [1, 2, 4, 4])),
        num_fields=num_fields,
        scale_shift=mcfg.get("scale_shift", False),
        num_res_blocks=mcfg.get("num_res_blocks", 2))

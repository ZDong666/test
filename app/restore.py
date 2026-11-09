"""封装上游 LanPaintBench 的真实推理：模型构建、管线复用、NFE 计数、计时与显存。

设计要点（见构建方案 §3/§4.4）：
- 不导入上游 ``run.py``（其在模块级执行 argparse 与相对路径加载），只引入所需模块。
- 复用上游 ``ULDCharaPipeline_Sampler``（LanPaint-K 管线）与 ``same_seeds``。
- 上游掩码极性为 1 = 保留，调用前用 ``mask_to_upstream`` 取反。
- NFE 通过包裹 ``unet.forward`` 实测计数（上游每次内层迭代会多次前向）。
"""

from __future__ import annotations

import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from app.data import mask_to_upstream

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_DIR = PROJECT_ROOT / "third_party" / "LanPaintBench"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "256x256_diffusion_uncond.pt"

_PIPELINE_CACHE: dict = {}


@dataclass(frozen=True)
class RestoreConfig:
    """单次修复的采样配置；默认值取上游 makefile 的官方参数。"""

    n_steps: int = 5  # 内层 Langevin 迭代次数 = 预算 K
    outer_steps: int = 20  # 外层扩散步数 T（num_inference_steps）
    step_size: float = 0.15
    alpha: float = 0.0
    friction: float = 15.0
    m: float = 1.0
    chara_lamb: float = 8.0
    chara_beta: float = 1.0
    seed: int = 42
    device: str = "cuda"
    checkpoint: Path = DEFAULT_CHECKPOINT


@dataclass
class RestoreResult:
    """一次修复的输出与开销证据。"""

    image: torch.Tensor  # (1, 3, 256, 256) float32，范围 [-1, 1]
    nfe: int
    elapsed_seconds: float
    peak_vram_mb: float
    cache_hit: bool = False
    seed: int = 42
    config: RestoreConfig | None = None


class _ForwardCounter:
    def __init__(self) -> None:
        self.value = 0


def _ensure_upstream_on_path() -> None:
    path = str(UPSTREAM_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)


def _build_pipeline(config: RestoreConfig, device: torch.device):
    """构建并缓存 (pipeline, counter, same_seeds)；模型权重从 config.checkpoint 加载。"""
    _ensure_upstream_on_path()
    from diffusers import (
        DDPMScheduler,
        DPMSolverMultistepScheduler,
        EulerDiscreteScheduler,
        UNet2DModel,
    )
    from diffusers.models.unets.unet_2d import UNet2DOutput
    from guided_diffusion.unet import UNetModel
    from pipeline_lib_imagenet import ULDCharaPipeline_Sampler, same_seeds

    checkpoint = Path(config.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"未找到模型权重：{checkpoint}")

    model = UNetModel(
        image_size=256,
        in_channels=3,
        model_channels=256,
        out_channels=6,  # learn_sigma=True
        num_res_blocks=2,
        attention_resolutions=(8, 16, 32),  # 上游 "32,16,8"
        dropout=0,
        channel_mult=(1, 1, 2, 2, 4, 4),
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=True,
        use_new_attention_order=False,
    )
    state_dict = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    noise_scheduler = DDPMScheduler(
        beta_start=0.0001,
        beta_end=0.02,
        num_train_timesteps=1000,
        beta_schedule="linear",
    )
    dpm_scheduler = DPMSolverMultistepScheduler.from_config(noise_scheduler.config)
    euler_scheduler = EulerDiscreteScheduler.from_config(noise_scheduler.config)

    class _ScoreNet(UNet2DModel):
        """与上游 run.py 中 score_net 等价的包装：取前 3 通道作为 eps。"""

        def __init__(self, model, scheduler, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.scheduler = scheduler
            self.unet_model = model

        def forward(self, x, timestep, encoder_hidden_states=None, return_dict=True):
            eps = self.unet_model(x, timestep)
            eps = eps[:, :3, :, :]
            if not return_dict:
                return (eps,)
            return UNet2DOutput(sample=eps)

    unet = _ScoreNet(
        model,
        dpm_scheduler,
        sample_size=256,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        norm_num_groups=1,
        block_out_channels=(1,),
        attention_head_dim=1,
        down_block_types=("DownBlock2D",),
        up_block_types=("AttnUpBlock2D",),
    )

    def cal_eps(self, image_rescale, t):
        return self.__call__(
            image_rescale,
            t.expand(image_rescale.shape[0]).to(image_rescale.device),
        ).sample

    unet.cal_eps = types.MethodType(cal_eps, unet)

    pipeline = ULDCharaPipeline_Sampler(
        unet=unet,
        scheduler=euler_scheduler,
        step_size=config.step_size,
        n_steps=config.n_steps,
        friction=config.friction,
        m=config.m,
        alpha=config.alpha,
        chara_lamb=config.chara_lamb,
        chara_beta=config.chara_beta,
    ).to(device)

    counter = _ForwardCounter()
    original_forward = unet.forward

    def counted_forward(self, x, timestep, *args, **kwargs):
        counter.value += 1
        return original_forward(x, timestep, *args, **kwargs)

    unet.forward = types.MethodType(counted_forward, unet)
    return pipeline, counter, same_seeds


def _get_pipeline(config: RestoreConfig, device: torch.device):
    key = (
        config.n_steps,
        config.step_size,
        config.alpha,
        config.friction,
        config.m,
        config.chara_lamb,
        config.chara_beta,
        str(device),
        str(Path(config.checkpoint)),
    )
    if key not in _PIPELINE_CACHE:
        _PIPELINE_CACHE[key] = _build_pipeline(config, device)
    return _PIPELINE_CACHE[key]


def restore(image: torch.Tensor, mask: torch.Tensor, config: RestoreConfig) -> RestoreResult:
    """对单张图片做一次真实 LanPaint 修复（batch=1）。

    ``image``：(1, 3, 256, 256)，[-1, 1]；``mask``：(1, 1, 256, 256)，{0, 1}，1 = 缺失。
    """
    device = torch.device(config.device)

    if float(mask.sum().item()) == 0.0:
        return RestoreResult(
            image=image.detach().clone(),
            nfe=0,
            elapsed_seconds=0.0,
            peak_vram_mb=0.0,
            cache_hit=False,
            seed=config.seed,
            config=config,
        )

    pipeline, counter, same_seeds = _get_pipeline(config, device)

    same_seeds(config.seed)
    counter.value = 0

    use_cuda = device.type == "cuda"
    y = image.to(device)
    upstream_mask = mask_to_upstream(mask).to(device)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.perf_counter()
    output = pipeline(
        y=y,
        mask=upstream_mask,
        generator=generator,
        num_inference_steps=config.outer_steps,
        output_type="numpy",
    )
    if use_cuda:
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    peak_vram_mb = (
        torch.cuda.max_memory_allocated() / (1024.0 ** 2) if use_cuda else 0.0
    )
    result_image = torch.from_numpy(np.ascontiguousarray(output.images)).to(torch.float32)

    return RestoreResult(
        image=result_image,
        nfe=counter.value,
        elapsed_seconds=elapsed,
        peak_vram_mb=float(peak_vram_mb),
        cache_hit=False,
        seed=config.seed,
        config=config,
    )

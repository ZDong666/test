# LanPaint 系统构建 —— S1 最小切片（下载 + 真实推理验证）

> 依据：`PRD_LanPaint.md` v1.0；项目根：`d:\Study\Postgraduate\人工智能项目`（用户确认以当前目录为根）
> 用户反馈：原全量计划范围太大，要求拆小；**先下载上游与权重并验证真实推理能跑通**，再写其余模块。
> 本文件只描述**立即执行的第一片（S1）**，后续切片见附录。

---

## 1. 摘要与判定标准

**S1 目标**：在 8 GB 显存的本机跑通一次**真实**的 LanPaint 256×256 图像修复，并留下耗时 / NFE / 峰值显存证据，为 PRD 的 D1/D2 门禁提供事实依据。S1 **不实现**预算策略、实验编排、离线评价、缓存与 Gradio 界面。

**判定标准（全部通过才算 S1 完成）**

1. `third_party/LanPaintBench` 存在且 `HEAD == ea5be5d94be40faf15396d3a4139f6ff26e63a2a`。
2. `checkpoints/256x256_diffusion_uncond.pt` 字节数 `== 2211383297`，并记录 SHA-256。
3. 至少 1 张 256×256 真实图片 + 1 个掩码可用于推理。
4. `K=5`、`T=20`、`batch=1`、FP32 下生成有效图像：无 NaN / Inf / OOM / 全空白。
5. 记录实测 NFE、单次耗时、峰值显存。
6. 连续 10 次成功率 `10/10`（D2 门禁），并给出 495 次实验的 GPU 小时估算。

---

## 2. 当前实际状态（本轮已核实，非假设）

| 项 | 状态 | 证据 |
|---|---|---|
| 项目根 git 仓库 | ✅ 已初始化 | `git rev-parse --is-inside-work-tree` → `true` |
| `.gitignore` | ✅ 已存在 | 已排除权重 / 上游克隆 / 数据 / 缓存 |
| `third_party/LanPaintBench` | ✅ 已克隆，工作区干净 | `HEAD = ea5be5d94be40faf15396d3a4139f6ff26e63a2a`；非 `.git` 文件 **23** 个 |
| `checkpoints/256x256_diffusion_uncond.pt` | ❌ **不完整** | 实际仅 **12288 字节**（应为 2211383297），需重下 |
| `data/samples/original`、`data/masks` | ⚠️ 目录已建、**内容为空** | 需 `prepare_samples.py` 填充 |
| `results/raw` | ⚠️ 目录已建、内容为空 | 需 `smoke_inference.py` 填充 |
| `app/` | ⚠️ 仅 `__init__.py` | `data.py` / `masks.py` / `restore.py` 待写 |
| `scripts/` | ❌ 不存在 | 待建 |
| Python 环境 | ✅ 可用 | `D:\Study\Postgraduate\GPT\envs\lanpaint\Scripts\python.exe` → Python 3.12.16 |
| 本机代理 `127.0.0.1:7890` | ✅ 可连通 | `Test-NetConnection` → `True` |
| D 盘剩余空间 | ✅ 314.2 GB | 足够放 2 GB 权重 |
| `third_party/LanPaintBench/guided_diffusion/` | ✅ 完整 | 含 `unet.py`、`nn.py`、`fp16_util.py`、`__init__.py` 等 13 个文件 |

> 结论：**上游与仓库基线已完成，权重未完成，应用代码未开始**。S1 剩余工作 = 补权重 + 写 5 个文件 + 跑验证。

---

## 3. 上游 API 实测契约（写代码前已确认，避免臆测）

来源：`third_party/LanPaintBench/run.py`、`pipeline_lib_imagenet.py`、`makefile`。

- **模型构造**（`run.py` `create_model`，参数由 `run.py` 的全局常量决定）：
  `image_size=256, in_channels=3, model_channels=256, out_channels=6(learn_sigma=True), num_res_blocks=2, channel_mult=(1,1,2,2,4,4), attention_resolutions=(8,16,32)[即 "32,16,8" 换算], num_head_channels=64, use_scale_shift_norm=True, resblock_updown=True, use_fp16=False, num_heads=1, dropout=0`。
  权重加载：`model.load_state_dict(torch.load('<ckpt>'))`（**与路径无关，必须自行传路径**）。
- **score_net 包装**（`run.py:342`）：`class score_net(UNet2DModel)`，`forward` 内 `eps = self.unet_model(x, timestep)[:, :3]`，返回 `UNet2DOutput(sample=eps)`；另用 `types.MethodType` 绑定 `cal_eps(self, image_rescale, t)`。
- **采样入口**（`ConditionPipeline.__call__`，`pipeline_lib_imagenet.py:120`）：
  ```python
  pipeline(y=y, mask=mask, generator=gen, num_inference_steps=T, output_type="numpy").images
  ```
  - `y`：观测图（缺失区域内容无意义）；`mask`：**1 = 已知/保留**；返回 `sample_loop` 尾部 `image*(1-mask) + y*mask`。
  - `num_inference_steps` ⇒ 外层扩散步数 `T`（`scheduler.set_timesteps(T)`）。
- **内层预算**（`ULDCharaPipeline_Sampler.__init__`，`pipeline_lib_imagenet.py:283`）：构造时传入的 `n_steps` ⇒ **内层 Langevin 迭代次数 K**；`sample_loop:215` 的 `for i in range(self.n_steps)` 即 K 次。
- **官方参数在 makefile 而非 argparse 默认值**（`makefile:3`）：
  `--method LanPaint-5 --ite_step_size 0.15 --uld_alpha 0 --uld_friction 15 --param_m 1 --sample_steps 20`
  ⇒ `step_size=0.15, alpha=0.0, friction=15, m=1, T=20`，管线用 `ULDCharaPipeline_Sampler`（`chara_lamb=8, chara_beta=1.0` 为 argparse 默认）。
- **调度器**：`euler_scheduler = EulerDiscreteScheduler.from_config(DDPMScheduler(beta_start=1e-4, beta_end=0.02, num_train_timesteps=1000, beta_schedule='linear').config)`。
- **NFE 不可用公式推断**：每次内层迭代会多次调用 `cal_eps/score_model`（ULDChara 的 `langevin_dynamics` 半步步进会二次求 `Coef_C`），故 **必须包裹 `unet.forward` 实测计数**。
- **随机性**：`pipeline_lib_imagenet.py` 导入即 `same_seeds(0)`；`y_forward_diffusion` 与 `torch.randn_like` 均使用全局 RNG ⇒ **每次推理前必须重设全局种子**。
- **输出数值范围**：`image.cpu().numpy()` 为 `(B,3,256,256)` 的 **[-1,1]** 张量（旁证：`run.py:91` `psnr` 中 `(img1+1)/2`）。
- **不导入 `run.py`**：其模块级会执行 `argparse`（`--method` 必填）并 `torch.load('./checkpoints/...')` 相对路径，作为库导入会直接崩溃。

---

## 4. 拟变更（文件清单 + 具体做法）

### 4.1 新增 / 修改文件总览

```
d:\Study\Postgraduate\人工智能项目\
├─ third_party\LanPaintBench\UPSTREAM.md   [新增] 来源、commit、许可、权重 SHA、未改源码声明
├─ checkpoints\256x256_diffusion_uncond.pt [修复] 重下至 2211383297 字节
├─ data\
│  ├─ samples\original\*.png               [生成] 10 张 ImageNet 256 图
│  ├─ masks\*.png                          [生成] 对应中心块掩码
│  └─ manifest.csv                         [生成] 最小样本清单
├─ results\raw\s1_smoke\
│  ├─ images\*.png                         [生成] 修复结果
│  └─ report.json                          [生成] D2 证据
├─ app\
│  ├─ __init__.py                          [已存在，不改]
│  ├─ data.py                              [新增]
│  ├─ masks.py                             [新增，仅 center]
│  └─ restore.py                           [新增，核心]
└─ scripts\
   ├─ prepare_samples.py                   [新增]
   └─ smoke_inference.py                   [新增]
```

**明确不建**（留给后续切片）：`config.py`、`budget.py`、`features.py`、`metrics.py`、`cache.py`、`experiment.py`、`ui.py`、`tests/`、`configs/`、`docs/`、CI、README。

### 4.2 `app/data.py`（I/O 与对齐）

职责：图片/掩码读写、256 对齐、`[-1,1]↔[0,1]` 转换、掩码极性转换、哈希。全部用 `pathlib`。

```python
MODEL_SIZE = 256

unit_to_model(x) -> Tensor      # [0,1] -> [-1,1]：x*2-1
model_to_unit(x) -> Tensor      # [-1,1] -> [0,1]：(x+1)/2

align_image(img: Image.Image, size=256) -> Image.Image   # 双三次 BICUBIC
align_mask(mask: Image.Image, size=256) -> Image.Image   # 最近邻 NEAREST

load_image(path: Path) -> Tensor   # (1,3,256,256) float32，[-1,1]；灰度复制为 3 通道
load_mask(path: Path)  -> Tensor   # (1,1,256,256) float32，{0,1}；二值化阈值 0.5
mask_to_upstream(mask: Tensor) -> Tensor   # 1 - mask（上游 1=保留）
save_result(path: Path, image: Tensor) -> None  # [-1,1] -> clip[0,1] -> uint8 PNG
sha256_file(path: Path, chunk=1<<20) -> str
```

约定：内部 `mask=1` 表示**缺失**（PRD）；传给上游前一律 `mask_to_upstream`。

### 4.3 `app/masks.py`（S1 仅 center）

```python
generate_mask(mask_type: str, size=256, target_area=0.30, seed=42) -> Tensor
    # 返回 (1,1,size,size) float32 {0,1}；S1 只实现 mask_type="center"
    # 其余类型 raise NotImplementedError（S4 补 划痕/零散块）
mask_area_ratio(mask: Tensor) -> float
save_mask_png(path: Path, mask: Tensor) -> None   # {0,1} -> uint8 *255
```

`center` 行为（确定性，seed 仅保留接口位）：
- `side = max(1, round(sqrt(target_area) * size))`；取居中正方形 `[ (size-side)//2 : (size-side)//2+side ]`，置 1。
- 实际面积与 `target_area` 偏差需 `<= 0.02`（±2pp），`prepare_samples.py` 记录 `actual_area`。

### 4.4 `app/restore.py`（核心，封装上游采样）

```python
@dataclass(frozen=True)
class RestoreConfig:
    n_steps: int = 5          # 内层预算 K
    outer_steps: int = 20     # T
    step_size: float = 0.15
    alpha: float = 0.0
    friction: float = 15.0
    m: float = 1.0
    chara_lamb: float = 8.0
    chara_beta: float = 1.0
    seed: int = 42
    device: str = "cuda"
    checkpoint: Path = Path("checkpoints/256x256_diffusion_uncond.pt")

@dataclass
class RestoreResult:
    image: Tensor             # (1,3,256,256) [-1,1]
    nfe: int
    elapsed_seconds: float
    peak_vram_mb: float
    cache_hit: bool = False
    seed: int = 42
    config: RestoreConfig = None

def restore(image: Tensor, mask: Tensor, config: RestoreConfig) -> RestoreResult
```

`restore` 内部步骤（严格按此实现）：
1. 将 `third_party/LanPaintBench` 注入 `sys.path`（幂等，仅当不存在时插入）。
2. **空掩码短路**：`mask.sum() == 0` ⇒ 直接返回原图，`nfe=0`、耗时为 0、`peak_vram_mb=0`。
3. 取进程级缓存（按 `(n_steps, device, checkpoint)` 键）复用 pipeline；未命中则构建：
   - `model = UNetModel(...)`（参数见第 3 节，`attention_resolutions=(8,16,32)`）。
   - `model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))`；失败记录并回退 `weights_only=True` 之外的方案。`model.eval()` 且 `.to(device)`、FP32。
   - 构造 `score_net(model, dpm_scheduler, sample_size=256, in_channels=3, out_channels=3, layers_per_block=1, norm_num_groups=1, block_out_channels=(1,), attention_head_dim=1, down_block_types=("DownBlock2D",), up_block_types=("AttnUpBlock2D",))`；`types.MethodType` 绑定 `cal_eps`。
   - `pipeline = ULDCharaPipeline_Sampler(unet=unet, scheduler=euler_scheduler, step_size=cfg.step_size, n_steps=cfg.n_steps, friction=cfg.friction, m=cfg.m, alpha=cfg.alpha, chara_lamb=cfg.chara_lamb, chara_beta=cfg.chara_beta).to(device)`。
4. 包裹 NFE 计数：把 `unet.forward` 换成带计数器的包装（保留原函数签名与返回值）。计数器在**每次调用 `restore` 前重置**。
5. 复现性：`same_seeds(config.seed)`（直接调用上游 `pipeline_lib_imagenet.same_seeds`）。
6. 计时与显存：`torch.cuda.synchronize(); reset_peak_memory_stats(); t0=time.perf_counter()` → 调用
   `pipeline(y=image, mask=mask_to_upstream(mask), generator=torch.Generator(device="cpu").manual_seed(config.seed), num_inference_steps=config.outer_steps, output_type="numpy")` →
   `torch.cuda.synchronize(); elapsed = time.perf_counter()-t0`；`peak_vram_mb = max_memory_allocated() / 1024**2`。
7. 结果转换：`torch.tensor(out.images, device=device)`（`(1,3,256,256)`，[-1,1]）。
8. 返回 `RestoreResult(...)`。

约束：`restore` 只做单图 `batch=1`；不做任何策略/预算决策（那是 S2/S5）。

### 4.5 `scripts/prepare_samples.py`

CLI：`--count 10`、`--target-area 0.30`、`--seed 42`、`--out data`。
- 设 `os.environ["HF_ENDPOINT"]="https://hf-mirror.com"`。
- `load_dataset("benjamin-paine/imagenet-1k-256x256", data_files={"test": "data/test-*.parquet"}, streaming=True, verification_mode="no_checks")` 取前 `count` 张。
- 每张：存 `data/samples/original/img{i:04d}.png`（256 RGB）→ `generate_mask("center", ...)` → `save_mask_png` 到 `data/masks/img{i:04d}_center{mask30}_mask.png`。
- 写 `data/manifest.csv`，字段：`sample_id, original_image_id, split, mask_type, target_area, actual_area, seed, image_path, mask_path, sha256`（PRD §7.2 子集；`split="check"`）。
- 失败兜底：HF 不可用时允许 `--local-dir` 指认本地 256 图片目录，并打印警告（记录到 stdout）。

### 4.6 `scripts/smoke_inference.py`

CLI：`--runs 1`、`--sample <png>`（默认取 `data/samples/original/` 第一张）、`--mask <png>`（默认取对应掩码）、`--n-steps 5`、`--outer-steps 20`。
- 逐次调用 `restore`，每次都 `torch.cuda.empty_cache()`。
- 每次判定：`nan_or_inf = not torch.isfinite(out).all()`；`blank = (out.std() < 1e-3)`。
- 保存结果图：`results/raw/s1_smoke/images/run{i:02d}.png`（`save_result`）。
- 写 `results/raw/s1_smoke/report.json`：
  ```json
  {
    "runs": 10, "ok_runs": 10,
    "nfe_per_run": 5, "nfe_total": 50,
    "nan_or_inf": false, "blank": false,
    "peak_vram_mb": 0.0,
    "median_seconds": 0.0, "mean_seconds": 0.0,
    "estimated_gpu_hours_495": 0.0,
    "config": {"n_steps": 5, "outer_steps": 20, "seed": 42, "precision": "fp32"},
    "notes": ""
  }
  ```
  其中 `peak_vram_mb` 取各次最大值；`estimated_gpu_hours_495 = median_seconds * 495 / 3600`。

### 4.7 `third_party/LanPaintBench/UPSTREAM.md`

内容：仓库 URL、`commit = ea5be5d94be40faf15396d3a4139f6ff26e63a2a`、LICENSE 名称（见上游 `LICENSE`）、**未修改任何上游源码**的声明、权重 URL 与 SHA-256、克隆/恢复命令（含代理用法）。该文件本身位于被 `.gitignore` 忽略的目录内，仅作本地溯源记录。

---

## 5. 执行步骤（顺序）

1. **补权重**：重下 `checkpoints/256x256_diffusion_uncond.pt`
   ```powershell
   curl.exe -L --retry 3 --retry-delay 5 -C - -o "checkpoints\256x256_diffusion_uncond.pt" `
     "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt"
   ```
   校验：字节数 `== 2211383297`；否则删除后重下。随后计算 SHA-256。
2. **写 `UPSTREAM.md`**（含步骤 1 得到的 SHA-256）。
3. **写 `app/data.py`、`app/masks.py`、`app/restore.py`**。
4. **写 `scripts/prepare_samples.py` 并运行**，得到图片 + 掩码 + `data/manifest.csv`。
5. **写 `scripts/smoke_inference.py`**。
6. **验证**（见第 6 节）；如失败按第 8 节处置。
7. **汇报**：给出 D2 判定（是否降档 `K=1/3/5`、495 次 GPU 小时估算），再决定是否展开 S2。

---

## 6. 验证步骤

```powershell
$py = 'D:\Study\Postgraduate\GPT\envs\lanpaint\Scripts\python.exe'
# 1) 环境与依赖仍无冲突
& $py tools\check_lanpaint_env.py
# 2) 准备样本（10 张图 + 10 个掩码）
& $py scripts\prepare_samples.py --count 10 --target-area 0.30
# 3) 单次真实推理
& $py scripts\smoke_inference.py --runs 1
# 4) 连续 10 次（D2 门禁）
& $py scripts\smoke_inference.py --runs 10
```

通过条件（读 `results/raw/s1_smoke/report.json`）：
`ok_runs == runs` 且 `nfe_per_run > 0` 且 `nan_or_inf == false` 且 `blank == false` 且 `peak_vram_mb < 8000`。
另需人工抽查 `results/raw/s1_smoke/images/run00.png`（中心区域被合理修复、边缘保持原作）。

---

## 7. 假设与决策

| 项 | 决策 | 依据 |
|---|---|---|
| 项目根 | 当前目录 `d:\Study\Postgraduate\人工智能项目` | 用户确认 |
| 网络 | 允许走本机代理 `127.0.0.1:7890`（git）；HF 走 `hf-mirror.com`；权重直连 Azure blob | 用户确认 + 实测 |
| S1 范围 | 只做「补权重 + 5 个文件 + 真实验证」 | 用户要求拆小 |
| 掩码极性 | 内部 `mask=1` 缺失；传上游 `1-mask` | 上游源码 `sample_loop` 尾部 |
| 采样参数 | 取 makefile 值（非 argparse 默认值） | `makefile:3` |
| 精度 | FP32（`use_fp16=False`） | PRD §2.3 / 上游 `run.py` |
| batch | 1 | PRD §8.2 `batch_size: 1`，8 GB 显存 |
| NFE | 实测计数 | 上游循环内多次前向 |
| 不安装 | `tensorflow`、`mpi4py`、`torch-fidelity` | PRD §4.2 |
| 上游源码 | 不改动，只读引用 | 便于复现与溯源 |

---

## 8. 风险与应对（S1）

| 风险 | 应对 |
|---|---|
| `torch.load` 在 torch 2.11 默认 `weights_only=True` 下加载上游权重失败 | 显式 `weights_only=False`；若仍失败记录完整报错并暂停 |
| FP32 在 8 GB 显存 OOM | `batch=1` + 每次 `empty_cache()`；仍 OOM 则记录并转 PRD §11.2 失败分析 |
| 上游为 Linux 编写（`makefile`） | 不调用 `make`；只用 Python 模块；路径全 `pathlib` |
| HF/HF-mirror 下载慢或失败 | 流式取用；失败则用 `--local-dir` 本地 256 图片兜底并记录 |
| 权重下载剩余 12 KB 残片导致校验失败 | 校验字节数；不符则删除后整份重下 |
| 结果数值范围判断错误（[0,1] vs [-1,1]） | 按 [-1,1] 处理并 `save_result` 裁剪；若图像全黑/全白则回查工厂换算 |

---

## 附录：后续切片（本次不做）

- **S2**：`features.py` + `budget.py` + 单元测试（离线可验证）。
- **S3**：`metrics.py` + `cache.py` + `config.py` + `scripts/evaluate.py`。
- **S4**：三类掩码（中心块/划痕/零散块）+ 15%/30% 面积控制 + 50 张清单。
- **S5**：`experiment.py` + `configs/*.yaml` + 批量等预算执行。
- **S6**：`ui.py`（Gradio）+ `start_ui.ps1`。
- **S7**：README、`docs/` 三份协作文档、CI。

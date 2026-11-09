# LanPaint 项目前置准备清单

编制日期：2026-11-09

依据文件：`AI项目实施计划_LanPaint主方案与工业检测备选.md`

本文只做启动前审计和准备安排，不替代计划。硬件和本机环境以本次实测为准；上游仓库、模型、数据和依赖是否可运行，仍需在联网后逐项验证。

## 1. 结论

当前状态可以概括为：**硬件与 PyTorch/CUDA 基础可用，但项目环境和实验材料尚未就绪。**

| 项目 | 状态 | 证据或缺口 |
|---|---|---|
| CPU、内存、磁盘 | 可用 | 24 个逻辑处理器、约 16 GB 内存、D 盘约 316.8 GB 空闲 |
| GPU | 可用 | RTX 5070 Laptop，8151 MiB 显存，驱动 610.88 |
| Python 3.12 与 PyTorch | 基础可用 | `D:\Study\Postgraduate\GPT\envs\mnist\python.exe` 为 Python 3.12.16、PyTorch 2.11.0+cu128 |
| CUDA 实际运算 | 通过 | PyTorch 识别 sm_120，256×256 GPU 矩阵乘法成功 |
| 独立项目环境 | 已安装，共享 PyTorch | `D:\Study\Postgraduate\GPT\envs\lanpaint`，Python 3.12.16，继承 `mnist` 的 PyTorch |
| LanPaint 依赖 | 已完成 | 核心、评价、实验和 Gradio 依赖已安装，`pip check` 无冲突 |
| 网络 | 阻塞 | 当前终端 DNS 解析 GitHub 失败，代理变量未设置 |
| 上游代码与权重 | 未完成 | 尚未克隆固定 commit，未准备 `256x256_diffusion_uncond.pt` |
| 数据与掩码 | 未完成 | 尚无固定 50 张图片清单、三类掩码生成器及数据哈希 |
| 视频与展示工具 | 部分完成 | Git、PowerShell、VS Code 可用；未发现 `ffmpeg` |
| 论文材料 | 未完成 | 尚无文献矩阵、符号表、相关工作对照表和实验协议 |

启动实验前必须通过第 9 节的 P0 门禁。尤其是先完成一次真实 LanPaint 推理，再决定是否进入 495–720 次生成实验；基础 CUDA 矩阵运算不能证明完整扩散模型可以稳定运行。

## 2. 本机实测资源

### 2.1 硬件与系统

- 操作系统：Windows 11，版本 10.0.26200。
- CPU：24 个逻辑处理器。
- 内存：约 16 GB。
- GPU：NVIDIA GeForce RTX 5070 Laptop GPU，8151 MiB 显存。
- NVIDIA 驱动：610.88；驱动报告的 CUDA UMD 版本为 13.3。
- 磁盘：C 盘约 126.5 GB 空闲，D 盘约 316.8 GB 空闲。

本机不需要为了 PyTorch 再单独安装完整 CUDA Toolkit。当前 PyTorch wheel 已带 CUDA 12.8 运行库并通过 GPU 运算；只有编译自定义 CUDA 扩展时才需要考虑 Toolkit 和 Visual Studio Build Tools。

### 2.2 Python 与深度学习框架

- 系统默认 `python`：Python 3.10.5，没有 PyTorch，不适合作为本项目环境。
- 实际可用的深度学习环境：`D:\Study\Postgraduate\GPT\envs\mnist\python.exe`。
- 该环境为 Python 3.12.16、pip 26.2.2。
- 已安装 `torch 2.11.0+cu128`、`torchvision 0.26.0+cu128`。
- 已安装 `numpy`、`Pillow`、`matplotlib`、`tqdm` 等少量基础包。
- `D:\Study\Postgraduate\GPT\envs\.wheels` 已缓存 PyTorch 与 torchvision 的 cp312 Windows wheel，可在无网络时复用于新环境。

虽然有可用基础环境，但 `mnist` 是既有训练环境。第一版可以用于快速验证 LanPaint 是否可加载；正式实验应使用单独的 `lanpaint` 环境，避免依赖冲突并生成可复现的锁文件。

`lanpaint` 是从 `mnist` 环境派生的嵌套虚拟环境，已通过 `include-system-site-packages = true` 共享 `mnist` 中的 PyTorch 和 torchvision。它自己的 `site-packages` 优先于共享环境，因此本项目锁定的 NumPy、Pillow、Matplotlib 等版本会覆盖共享环境中的同名包。必须保留 `mnist`，不能删除或破坏其 PyTorch 安装；后续获得独立 Python 3.12 安装包后，可再用官方解释器建立完全独立的环境。

### 2.3 当前工具状态

| 工具 | 状态 | 用途 |
|---|---|---|
| Git | 可用 | 克隆、版本固定、提交和推送 |
| PowerShell 7 | 可用 | 本地启动和自动检查 |
| VS Code | 可用 | 开发、调试和查看结果 |
| curl | 可用 | 网络与下载连通性检查 |
| pandoc | 可用 | Markdown 转文档，可选 |
| ffmpeg | 未发现 | 演示视频、图片序列转视频 |
| GitHub CLI | 未发现 | 可选，不是必需 |
| conda 启动器 | 当前异常 | `conda.exe` 报系统 DLL 重定位错误，不作为本项目主入口 |

## 3. 目标环境设计

### 3.1 目录建议

```text
D:\Study\Postgraduate\GPT\
├─ envs\lanpaint\                 # 独立 Python 环境
├─ ai_inpainting_project\         # ZDong666/test 的独立克隆
├─ models\lanpaint\               # 权重，不提交 Git
├─ data\lanpaint\                 # 固定图片、掩码和清单，按许可处理
├─ cache\lanpaint\                # 推理缓存，不提交 Git
└─ outputs\lanpaint\              # 实验输出，只提交摘要和小样例
```

当前工作区不是最终项目仓库。开始开发时应先检查远端 `ZDong666/test` 的默认分支和已有内容，再在独立目录克隆；不能把父目录的学习代码整体提交。

### 3.2 环境创建策略

推荐在联网或本地 wheel 可用时执行，具体上游版本在克隆代码后确认：

```powershell
$base = 'D:\Study\Postgraduate\GPT\envs\mnist\python.exe'
$envDir = 'D:\Study\Postgraduate\GPT\envs\lanpaint'
& $base -m venv $envDir
& "$envDir\Scripts\python.exe" -m pip install --upgrade pip
& "$envDir\Scripts\python.exe" -m pip check
```

随后先安装已有本地 torch wheel，再安装上游 `requirements.txt`：

```powershell
$py = 'D:\Study\Postgraduate\GPT\envs\lanpaint\Scripts\python.exe'
$wheels = 'D:\Study\Postgraduate\GPT\envs\.wheels'
& $py -m pip install "$wheels\torch-2.11.0+cu128-cp312-cp312-win_amd64.whl"
& $py -m pip install "$wheels\torchvision-0.26.0+cu128-cp312-cp312-win_amd64.whl"
```

不要直接在当前环境无版本约束地安装“最新版”。先读取上游 `requirements.txt`，记录实际 Python、包版本、安装来源和 wheel 哈希，再导出：

```powershell
& $py -m pip freeze > requirements-lock.txt
& $py -m pip check
```

### 3.3 依赖分层

最终清单必须以 LanPaintBench 实际 `requirements.txt` 为准。以下是按项目模块整理的核对清单，不是已经验证的安装命令。

| 类别 | 候选依赖 | 用途 | 当前状态 |
|---|---|---|---|
| 基础张量 | `torch`、`torchvision` | 模型与 GPU 推理 | 已安装 |
| 扩散模型 | `diffusers`、`transformers`、`accelerate`、`safetensors`、`huggingface_hub` | 模型、调度器和权重加载 | 未安装 |
| 图像与数值 | `numpy`、`Pillow`、`scipy`、`scikit-image` | 图像 I/O、掩码、数值和图像处理 | 部分安装 |
| 指标 | `lpips` | 主评价指标 LPIPS | 未安装 |
| 实验表格 | `pandas`、`matplotlib`、`tqdm`、`PyYAML` | 清单、CSV、图表、配置和进度 | 部分安装 |
| 交互界面 | `gradio` | 本地上传、绘制掩码、结果展示 | 未安装 |
| 测试与质量 | `pytest`、`ruff` | 本地检查和 CPU CI | 未安装 |
| 可选记录 | `tensorboard` | 损失、耗时和调试可视化 | 未安装 |
| 备选方案 | `anomalib` | EfficientAD 工业检测 | 未安装，且不应装入主环境 |

计划中的 `diffusers 0.34.0` 是待核对值。上游依赖、Python 3.12、NumPy/Pillow 新版本与 `lpips` 的兼容性需要在独立环境中一次装好并锁定。

以下内容第一版不要安装或启用：

- `xformers`、`bitsandbytes`：当前 256×256、batch=1 没有明确必要，Windows 兼容性也会增加变量。
- FID/Inception 依赖：第一版已经明确删除 FID。
- EfficientAD/Anomalib：只有触发备选路线时才建立独立环境。
- Docker、WSL、完整 CUDA Toolkit：不是当前主路线的前提；只有 Windows 原生适配失败时才评估 WSL2。

## 4. 必须准备的软件与材料

### 4.1 项目软件

- Python 3.12 独立虚拟环境。
- PyTorch 2.11.0+cu128 和 torchvision 0.26.0+cu128。
- LanPaintBench 固定 commit 对应依赖。
- Git、PowerShell、VS Code。
- 可用的网络或代理，用于首次获取 GitHub 代码、模型、数据、预训练指标权重和 Python 包。
- 可选 `ffmpeg`，用于 3–5 分钟演示视频。
- 可选 GitHub CLI，用于查看 Pull Request 和 Actions；普通 Git 命令已足够。

### 4.2 模型、数据和权重

- 上游 `scraed/LanPaint` 与 `scraed/LanPaintBench` 的固定 commit。
- `256x256_diffusion_uncond.pt`，下载后记录文件大小和 SHA-256。
- 固定 50 张原始图片清单：检查集 10、开发集 10、正式测试 30。
- 三类掩码生成脚本及其随机种子：中心块、细长划痕、零散小块。
- 约 15% 与 30% 两种目标缺失比例的实际面积记录。
- LPIPS 所需的预训练网络权重，提前缓存并验证断网可用。
- 至少 20 GB 的可用空间，给环境、权重、图片、生成结果、缓存和演示视频留余量。

### 4.3 论文与交付软件

- Zotero 或同类文献管理工具，用于统一题名、年份、DOI、引用键和阅读笔记。
- Word/WPS 或 LaTeX；如果采用 LaTeX，需要提前确认中文模板、字体和参考文献格式。
- PowerPoint/WPS 演示文稿，用于最终汇报。
- Python 绘图：`matplotlib` 为主，`pandas` 整理逐样本结果。
- Markdown 编辑器或 VS Code，用于协作文档和实验记录。

## 5. 模型与环境验证顺序

不要先下载完整数据或开始批量实验。按以下顺序逐级验证：

1. **网络与仓库**：能够解析并访问 GitHub，记录目标分支、上游 commit 和许可证。
2. **环境隔离**：新环境可启动，`python -m pip check` 无冲突。
3. **导入检查**：`torch`、`torchvision`、`diffusers`、`lpips`、`gradio` 等可导入。
4. **GPU 检查**：`torch.cuda.is_available()` 为真，设备为 RTX 5070 Laptop，256×256 运算成功。
5. **权重检查**：模型能够加载，记录大小、SHA-256、加载耗时和峰值显存。
6. **单样本真实推理**：256×256、batch=1，能看到有效修复结果。
7. **连续稳定性**：连续 10 次无 OOM、NaN、Inf 和明显空白输出。
8. **性能基线**：热启动 LanPaint-5 中位耗时、峰值显存和实际 NFE。
9. **成本决策**：按单次耗时估算 495 或 720 次任务的 GPU 小时数；最小规模超过 24 GPU 小时时先降档或触发备选评估。

本工作区提供了 `tools/check_lanpaint_env.py` 做前四步和基础工具检查。它不会下载或修改任何内容。

## 6. 实验开始前的冻结项

以下内容应在查看正式测试结果前固定，并写入实验配置：

- 上游仓库 URL、commit SHA、许可证与本地修改说明。
- 模型权重名称、来源、大小、SHA-256。
- Python 与全部依赖版本锁文件。
- 50 张图片的来源、ID、许可证和哈希，以及按原图划分的集合。
- 三类掩码生成参数、目标/实际面积、边界长度和随机种子。
- 外层步数 `T=20`、预算档位、精度、batch size、随机种子。
- 评价指标实现、裁剪规则、全图与局部指标定义。
- 缓存键字段和失效规则。
- 主指标选择规则、候选权重顺序和测试集使用范围。
- 是否启用提前停止、动态调度或缓存；所有对照必须一致。

## 7. 论文阅读路线

### 7.1 必读与选读

| 优先级 | 文献 | 阅读范围 | 读完必须产出 |
|---|---|---|---|
| P0 | LanPaint, TMLR 2025 | 摘要、问题定义、方法、实验设置；对照 LanPaintBench 采样代码 | 公式与代码变量映射表；明确内层迭代和预算含义 |
| P0 | RePaint, CVPR 2022 | 扩散修复、重采样和条件融合部分 | 一页流程图；说明 LanPaint 与其区别 |
| P0 | StrDiffusion, CVPR 2024 | 语义差异、结构一致性、重采样机制 | 与“按图分配预算”的差异表 |
| P1 | DDPM, NeurIPS 2020 | 前向/逆向过程和采样基础 | 论文背景公式与符号表 |
| P1 | DING, ICLR 2026 | 摘要、引言、高效修复思想 | 近期高效修复路线的一段综述 |
| P1 | LPIPS, CVPR 2018 | 指标定义和适用范围 | 指标定义、参考图要求和解释限制 |
| P2 | EfficientAD, WACV 2024 | 摘要、方法、异常分数和实验设置 | 仅在切换备选路线时完成 |
| P2 | Anomalib EfficientAD 文档 | 数据格式、训练/测试接口 | 仅在切换备选路线时完成 |

PSNR、SSIM 如果出现在论文中，应引用原始定义，并明确本项目的计算范围。生成式修复可能有多个合理答案，PSNR/LPIPS 不能单独证明视觉质量，必须配合局部图和失败案例分析。

### 7.2 文献阅读产出

第一阶段不要只做“读过”标记，至少形成以下四项：

1. **文献矩阵**：列包括文献、问题、核心方法、是否需要额外模型、与本项目关系、可复现内容、引用键。
2. **符号与代码映射**：把论文中的时间步、噪声、条件、重采样、迭代次数与 LanPaintBench 函数和变量逐项对应。
3. **相关工作对照表**：RePaint、StrDiffusion、LanPaint、DING 各自解决什么问题，本项目只改变了什么。
4. **可证伪假设**：写清楚什么结果支持预算分配有效，什么结果说明无效或只对特定掩码有效。

建议为每篇核心论文记录以下问题：

- 输入、输出和条件是什么？
- 每一次采样调用对应多少网络前向？
- 重采样或额外迭代发生在哪一层？
- 基线是否在相同 NFE、相同权重和相同随机种子下比较？
- 数据集、图片尺寸和评价指标是什么？
- 哪些机制已经存在，不能包装成本项目新增贡献？

### 7.3 论文写作框架

报告按以下顺序准备，每节先搭证据，再写解释：

1. 问题与背景：扩散修复的计算成本，为什么“不同掩码需要不同预算”值得验证。
2. 相关工作：DDPM/RePaint 基础、LanPaint 的 Langevin 内层迭代、StrDiffusion 的结构重采样、DING 的高效路线。
3. 方法：LanPaint 基线、面积与归一化边界特征、排序与等预算分档、单图阈值模式。
4. 实验设置：数据、三类掩码、面积控制、对照、种子、NFE、缓存和公平性。
5. 结果：质量、效率、显存、稳定性、质量—成本曲线和逐掩码分析。
6. 失败案例：结构断裂、纹理不连续、语义错误、分配错误和增加迭代无收益。
7. 局限与结论：小样本、启发式规则、过拟合风险、平台差异和后续工作。

论文中的贡献应表述为“小规模复现 + 一个受限、可解释的等预算分配实验”，不要在没有外部验证时宣称普遍性的新算法或全面提升。

## 8. 建议的前置准备顺序

### P0：必须先完成

- 解决 GitHub、PyPI 和权重下载的网络连通性，记录代理或镜像配置，不把凭据提交仓库。
- 检查 `ZDong666/test` 当前分支和内容，在独立目录克隆。
- 固定 LanPaintBench commit，复制并核对官方依赖、模型获取方式和许可证。
- 创建 `lanpaint` 独立环境，锁定依赖并运行 `pip check`。
- 下载权重和 LPIPS 权重，记录 SHA-256，断网执行一次导入与加载检查。
- 完成一张小掩码、一张中等掩码的真实 GPU 推理。
- 连续运行 10 次并记录耗时、峰值显存、NFE 和异常。
- 固定数据清单、掩码种子、划分和首轮基线配置。
- 创建协作文档：`docs/项目进度.md`、`docs/下一步计划.md`、`docs/队友改进意见.md`。

### P1：基线跑通后完成

- 生成三类掩码和 15%/30% 面积扩展。
- 实现缓存、逐样本日志、配置快照和恢复运行。
- 实现 PSNR、全图 LPIPS 和边界局部指标。
- 搭建最小 Gradio 界面，先接检查集中的固定样例。
- 完成 LanPaint、RePaint、StrDiffusion 的对照表。
- 形成论文方法、实验协议和图表骨架。

### P2：仅在对应条件出现时完成

- 安装 `ffmpeg` 并录制演示视频。
- 增加 GitHub Actions CPU 检查。
- 阅读 DING 和更多高效率扩散工作。
- 触发备选路线后，建立独立 Anomalib 环境并准备 MVTec AD。

## 9. 启动门禁

全部 P0 满足后才进入正式代码开发和批量生成：

- [ ] GitHub 可访问，目标仓库已核对并克隆。
- [ ] 上游 commit、许可证和官方依赖已固定。
- [ ] `lanpaint` 环境可启动，依赖冲突为 0。
- [ ] PyTorch 能调用 RTX 5070 Laptop，CUDA 运算通过。
- [ ] 模型和 LPIPS 权重已缓存，大小与哈希已记录。
- [ ] 至少一张真实修复图成功生成。
- [ ] 连续 10 次推理无致命错误。
- [ ] 单次耗时和显存已实测，实验规模可估算。
- [ ] 50 张图片清单和三类掩码方案已冻结。
- [ ] 三份协作文档已建立。
- [ ] LanPaint 论文与代码映射表已完成第一版。

当前最先需要解决的阻塞是网络。没有代码、权重和数据，继续安装其余 Python 包无法完成真实可行性验证。

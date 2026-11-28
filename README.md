# LanPaint 图像修复

基于现有 LanPaint 模型的本地修复原型与推理预算分配研究。基础界面、开发集冻结、正式主测试、面积扩展、三种子复核与失败分析均已完成，结论与交付物见[最终实验报告](docs/最终实验报告.md)。

## 启动界面

在项目目录的 PowerShell 中运行：

```powershell
.\start_ui.ps1
```

`start_ui.ps1` 使用本机固定 Python 路径；换电脑时，在准备好的环境中执行 `python -m app.ui`。环境选择和 CPU 页面验证步骤见[使用说明第 2 节](docs/使用说明.md)。当前 `requirements.yml` 不是历史实验的锁定环境，不应直接视为已验证的复现配置。
终端显示本地地址后，在浏览器中打开该地址（通常为 `http://127.0.0.1:7860`）。关闭服务可在终端按 Ctrl+C。

打开页面和生成区域预览不需要 GPU；非空区域的真实修复需要已准备的 `third_party/LanPaintBench` 上游源码、`checkpoints/256x256_diffusion_uncond.pt` 权重与可用 NVIDIA GPU。恢复说明见[上游溯源](docs/上游溯源.md)。

网页端为原生 H5 + FastAPI，不再依赖 Gradio。上传图片后，用画笔涂抹待修复区域，可用橡皮、撤销和清空调整标记；也可上传黑白掩码（白色修复、黑色保留）。点击“预览修复区域”确认，再点击“开始修复”。默认 K=5，输出为 256×256。完成后可通过滑块对比遮挡输入与修复结果，下载结果与配置；产物同时保存至 `results/ui/`。

GPU 请求串行排队；尚不支持运行中取消或自适应预算。无掩码时返回原图，全缺失掩码会提示重新选择。上传图片不展示需要参考答案的质量指标。真实 GPU 校验入口仍为 `python scripts/check_ui.py`，改为通过本地 HTTP API 调用。

完整操作步骤（界面、批量实验、评价、排错）见[使用说明](docs/使用说明.md)。

项目协作入口：[项目进度](docs/项目进度.md)、[下一步计划](docs/下一步计划.md)、[队友改进意见](docs/队友改进意见.md)。

## 当前验证状态

全量离线测试通过（`python -m pytest tests -q`），`ruff` 静态检查通过；离线 CPU 检查入口 `python ci/run_cpu_tests.py` 会屏蔽 CUDA、禁止联网。GitHub Actions 的 CPU 工作流见 `.github/workflows/cpu.yml`（不下载模型、不跑扩散推理）。2026-11-17 另在仓库外新目录从远端克隆，按当时的 `ci/requirements.txt` 重建环境，跑通全量测试、真实推理、一组实验与界面验证，见[复现记录](docs/复现记录.md)。

早期 Gradio 基础界面已通过一次真实 GPU API 修复与下载验证：K=5、NFE=120、采样约 40.05 秒、峰值显存约 2730 MB。H5 + FastAPI 版本已完成 CPU 回归（133 passed）与真实 GPU HTTP 校验：`python scripts/check_ui.py` 于 2026-11-27 通过，K=5、NFE=120、30.93 秒、峰值 2730 MB、修复面积 29.91%，结果图与配置可经 `/runs/<run_id>/` 下载。2026-11-16 完整开发集与正式实验均已完成，冻结、输入哈希与等预算核验通过。

完整开发集现有 **10 张原图、60 个掩码组合、180 条档位结果，0 失败**，六策略共 360 条记录。M1 在与 B2 相同总 NFE=7200 下，开发集 LPIPS 从 0.088518 降至 0.085453（降低 **3.46%**），但 PSNR 与边界 MAE 略差。

正式验证已于 **2026-11-16 23:11（北京时间）** 完成：30 张测试原图、10 张共同扩展子集，主测试加面积扩展和两次新增种子共 **540 次生成、1080 条策略记录，0 失败**，四阶段验收通过。主测试相同 NFE=10800 下，M1 LPIPS 为 0.119657，B2 为 0.117012，M1 均值更差 2.26%，配对差值区间跨零；15% 面积子集有改善，30% 面积三种子合并则退化。**因此保留固定 K=5 为默认策略，M1 只作为已评测的实验策略归档。**

完整结论、10 组案例（含 5 组退化）、失败原因、局限与 PRD 验收对照见 **[最终实验报告](docs/最终实验报告.md)**，汇报材料见 **[项目汇报 PPT](docs/LanPaint_项目汇报.pptx)**；统计细节见[正式实验报告](results/metrics/formal_campaign_20261116/formal_campaign_report.md)，案例明细见[预算诊断报告](results/analysis/budget_diagnosis_20261117/diagnosis.md)。后续安排见[下一步计划](docs/下一步计划.md)。

## 交付物索引

| 交付物 | 位置 |
|---|---|
| 最终实验报告 | [docs/最终实验报告.md](docs/最终实验报告.md) |
| 汇报 PPT（14 页） | [docs/LanPaint_项目汇报.pptx](docs/LanPaint_项目汇报.pptx) |
| 六策略统计与配对分析 | [results/metrics/formal_campaign_20261116](results/metrics/formal_campaign_20261116/formal_campaign_report.md) |
| 10 组案例与预算诊断 | [results/analysis/budget_diagnosis_20261117](results/analysis/budget_diagnosis_20261117/diagnosis.md) |
| 开发集分析与冻结配置 | [results/metrics/dev_budget](results/metrics/dev_budget/seed_42/20261116T084403334123Z_1d7f3deb/development_analysis.md) |
| 运行前检查证据 | `results/validation/formal_preflight_20261116/` |
| 干净克隆复现记录 | [docs/复现记录.md](docs/复现记录.md) |
| 使用说明（界面/实验/评价/排错） | [docs/使用说明.md](docs/使用说明.md) |
| 测试覆盖率报告（83%） | [docs/覆盖率报告.md](docs/覆盖率报告.md) |
| 离线 CPU 检查 | `ci/run_cpu_tests.py`、`.github/workflows/cpu.yml` |

## 批量实验可靠性更新（2026-11-11）

2026-11-17 已补齐[预算诊断与案例报告](results/analysis/budget_diagnosis_20261117/diagnosis.md)：10 组改善/退化案例、边界放大及分组预算收益分析。主测试中心增预算的平均收益不足以抵消划痕降预算损失；部分指标改善仍伴随语义错误。离线复现入口为 `python analysis/diagnose_budget.py`（加 `--cases-per-side 5` 生成 10 组案例），不重新推理或拟合策略。

结果目录改为 `results/raw/<experiment_id>/seed_<seed>/<run_id>/`，指标采用对应的 `results/metrics/` 目录；重跑保留旧日志并复用匹配缓存。运行报告给出实际输出路径。

面积扩展和重复实验需用 `--baseline-run` 指向已完成的种子 42 主测试输出目录。两者使用主测试预先固定的同一子集：20 图主测试取前 5 图，30 图取前 10 图；命令示例见 `scripts/run_experiment.py`。报告的 `comparison_complete` 表示样本结果完整，`equal_nfe` 表示完整集合上的 B2/M1/A1/C1 实际累计 NFE 相等。

归一化与规则冻结流程已实现：完整开发集运行使用 `--freeze-development --warmup`，成功后在该次运行目录生成 `frozen_policy.json`；测试集必须传入 `--frozen-policy <路径>`。加载时核对源码、权重和采样配置，拒绝开发集图片泄漏。开发集选择结果不代替正式测试结论。

使用 `python scripts/analyze_development.py --run-dir <开发集运行目录>` 导出预算质量图、逐样本收益表和分析 Markdown。

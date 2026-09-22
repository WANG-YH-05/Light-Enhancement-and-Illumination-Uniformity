# Reference-Theta RRNet

该分支用于“选择一个参会者的光照，使其他参会者向该光照统一”。它是基于
RRNet 的工程扩展，不属于原论文公开结构。

## 结构

训练样本为三元组：

```text
source：人物 A 的待增强图像
reference：人物 B 的参考光照图像
target：保持人物 A 身份和色度、采用 B 对应光照的真值
```

`ReferenceThetaRRNet` 使用共享 LPRM 分别编码 source 和 reference，再由轻量
条件器预测归一化光照参数修正量：

```text
theta = denormalize(theta_source + delta_theta(source, reference))
```

最终仍由 RRNet 的深度感知渲染器生成结果。模型不包含稠密 RGB 残差或
Delta-Y 残差。参考图在编码前转为灰度，渲染照明默认转换为中性三通道，
因此参考人物的肤色不会复制到输入人物。

训练继续使用现有 Mask：

- `relight_mask` 限制人物增强范围并保护背景；
- `skin_mask` 仅参与皮肤重建与色度损失；
- 推理仍由 MediaPipe 生成整个人像 Mask，并沿用现有边缘及时间平滑。

## 从头训练

不要添加 `--resume` 或 `--init-from`：

```powershell
Set-Location "E:\Lighting Enhancement Project\RRNet\Model"
& "E:\conda_envs\iclight\python.exe" train.py `
  --config "configs\rrnet_mead_reference_theta.yaml"
```

每次训练保存到：

```text
outputs/rrnet_mead_reference_theta/run_YYYYMMDD_HHMMSS/
```

## 视频推理

```powershell
& "E:\conda_envs\iclight\python.exe" infer_reference_theta_video.py `
  --config "configs\rrnet_mead_reference_theta.yaml" `
  --checkpoint "<checkpoint.pt>" `
  --input "<input.mp4>" `
  --reference "<reference.png>" `
  --output "<output.mp4>" `
  --light-every 10 `
  --depth-every 3
```

固定参考图只编码一次。视频中仅对 `theta` 做 RRNet 论文公式 (5) 对应的 EMA，
不存在额外的稠密残差时序状态。

## Relative-light 正式改进分支

`reference_theta` 直接预测最终光照参数，不能保证先消除每个输入原有的光照。
正式改进分支使用 `task: reference_relative`：

```text
source -> shared LPRM -> theta_in  -> L_in(source depth)
reference -> shared LPRM -> theta_ref -> L_ref(source depth)
output = source * clamp(L_ref / max(L_in, floor), min_gain, max_gain)
```

训练额外使用 `source_clean` 和 `reference_clean` 构造真实照明比，分别监督
`L_in`、参考图自身的 `L_ref`，以及参考光在输入人物深度上的目标照明图。
参考图和输入图在进入 LPRM 前均转为灰度，因此不传递参考人物肤色。

旧的 `mead_v2_masked_theta_stats_1000.npz` 是“坏光图到 clean 图”的校正
参数统计，方向与绝对照明估计相反，不能复用。首次训练前先执行：

```powershell
Set-Location "E:\Lighting Enhancement Project\RRNet\Model"
& "E:\conda_envs\iclight\python.exe" calibrate_relative_lighting_stats.py `
  --config "configs\rrnet_mead_reference_relative.yaml" `
  --samples 1000 `
  --batch-size 8 `
  --optimization-steps 100 `
  --output "checkpoints\mead_relative_theta_stats_1000.npz"
```

随后从头训练，不添加 `--resume` 或 `--init-from`：

```powershell
& "E:\conda_envs\iclight\python.exe" train.py `
  --config "configs\rrnet_mead_reference_relative.yaml"
```

训练输出位于：

```text
outputs/rrnet_mead_reference_relative/run_YYYYMMDD_HHMMSS/
```

视频推理入口为：

```powershell
& "E:\conda_envs\iclight\python.exe" infer_reference_relative_video.py `
  --config "configs\rrnet_mead_reference_relative.yaml" `
  --checkpoint "<checkpoint.pt>" `
  --input "<input.mp4>" `
  --reference "<reference.png>" `
  --output "<output.mp4>" `
  --light-every 10 `
  --depth-every 3
```

相对光照训练日志新增：

- `Lin`：输入绝对照明监督误差；
- `Lref`：参考绝对照明监督误差；
- `Ltgt`：参考光在输入人物深度上的照明监督误差；
- `gain`：当前相对光照增益均值；
- `theta_gap`：输入与参考归一化光照参数的平均距离。

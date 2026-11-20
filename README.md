# audio-clip-repair

## 题面

老录音修复一开，削波段被修成硬邦邦的方波：只按单样本阈值判削波，噪声尖峰被当削波；平顶判定不做长度下限，1 个样本也算；重建用线性插值，长区间出现明显凹陷；重建后端点不连续，接缝咔哒；软限幅用硬截断，谐波失真；增益恢复按全段均值，动态被压平；DC 偏移把静音段也算进去，估计偏 0.2；区间重叠不合并，重复重建；批量共用重建缓冲，轨道之间串；全段削波时不报错，输出静音。repair/clipping.py 是核心，tests/test_clipping.py 15个红。要修的是：…

## 运行方式

在仓库根目录运行：

```powershell
cd D:\gsb-20261119\audio-clip-repair
python -m pytest tests -q
```

初始失败环境必须恰好是：

```text
15 failed
```

输出尾部应为 `15 failed in ...s`，且没有 collection error、import error 或 syntax error。只需要 Python 3 和 pytest；全部离线，无外部依赖、无随机数、无 sleep。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## 验收基线

`tests/test_clipping.py` 是验收基线，不得修改、删除或弱化其中任何测试。实现只改 `repair/clipping.py`，保持 8 个公开函数的签名、样本范围 `[-1, 1]`、阈值与 `1e-6` 绝对容差契约，沿用 `ClipRepairError` 语义。音频一律使用 `list[float]`，全部离线、无随机、不 sleep。

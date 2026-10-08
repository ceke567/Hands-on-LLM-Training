# 第 1 周实验记录｜从零训练 Mini GPT

状态：第 1 周全部 4 次实验与交付已圆满完成！操作步骤以 Obsidian 中的第 1 周指南为准，本页记录真实执行结果。

## 第 1 次前置练习：理解训练循环（线性模型）

- **练习脚本**：[first_train.py](first_train.py)
- **目标任务**：拟合 $y = 3x + 2$，跑通 PyTorch 训练循环并在超算 BW 卡上验证环境。
- **训练 Loss 记录**：
  - `step   0`: `3.04770899`
  - `step  50`: `0.00160085`
  - `step 100`: `0.00000153`
  - `step 150`: `0.00000000`
  - `step 200`: `0.00000000`
- **收敛参数**：`weight = 3.0000`, `bias = 2.0000`（精确收敛至目标函数）。
- **训练机制核心总结**：
  1. **为什么清梯度**：PyTorch 默认累加梯度，清零是为了消除上一轮历史残余梯度，防止干扰本轮更新。
  2. **反向传播计算什么**：从 loss 倒推，计算出到底哪些参数有偏差及该偏差对 loss 的影响程度（即各参数的梯度）。
  3. **优化器更新了什么**：根据计算得到的梯度，真正更新模型的可学习参数（`weight` 和 `bias`）。


## 目标

- 跑通文本 → token → Decoder-only Transformer → loss → 反向传播 → 生成的完整流程。
- 训练一个约 10M–50M 参数的小模型，先以约 10M 为目标。
- 保存并重新加载 checkpoint，对比训练前后的生成结果。

## 数据与环境

| 项目 | 实际记录 |
| --- | --- |
| 数据来源与版本 | Tiny Shakespeare (`input.txt`, 1,115,394 字符) |
| 字符/token 表大小 | 65 (字符级 Tokenizer，无损编解码已验证) |
| 训练/验证划分 | 90% 训练集 (1,003,854 tokens) / 10% 验证集 (111,540 tokens) |
| 样本核对 | 3 组抽查移位样本核对已通过（输入与目标严格错开一位） |
| 加速卡与软件版本 | 见 [环境记录](../../docs/environment.md)，实验时再次核对 |

## 第 3 次练习：Mini GPT 模型搭建与前后向核验

- **脚本文件**：[minigpt.py](minigpt.py)
- **模型配置**：6 层 Decoder-only Transformer, `n_embd=384`, `n_head=6`, `block_size=128`, `vocab_size=65`
- **总参数量**：`10,746,624` (约 **10.75M** 参数，精准符合 10M 目标)
- **张量形状流动**：输入 `(4, 8)` -> 嵌入与注意处理 `(4, 8, 384)` -> 输出 Logits `(4, 8, 65)`
- **初始未训练 Loss**：`4.4488`（理论期望值 $-\ln(1/65) \approx 4.1744$）
- **反向传播验证**：通过（全部参数梯度计算正常，GPU 上前向与反向计算完全畅通）



## 实验配置

| 实验 ID | 参数量 | 上下文长度 | 单次批大小 | 学习率 | 训练步数 | 随机种子 | 与基线相比改动了什么 |
| --- | ---: | ---: | ---: | --- | ---: | ---: | --- |
| W01-01 | 10.75M | 128 | 32 | 3e-4 | 1500 | 42 | 初始基线 (6层, 384维, 6头) |

## 结果

| 实验 ID | 起始/结束训练 loss | 起始/结束验证 loss | 峰值显存 | token/秒 | 耗时 | checkpoint 保存位置 |
| --- | --- | --- | --- | --- | --- | --- |
| W01-01 | 4.2676 / 1.4489 | 4.2793 / 1.6477 | 1011.9 MB | 188,029 | 32.68s | `checkpoints/minigpt_final.pt` |

## 训练前后生成对比

固定提示词 `\n`、生成长度与 Multinomial 采样设置：

| 项目 | 内容 |
| --- | --- |
| 提示词与生成设置 | Prompt=`\n`, max_new_tokens=250, Multinomial 采样 |
| 训练前样例 | `"\nTCXM?Rpr;Zxmn;EopDgJpYXkGZv:bY$QttGr!rm!IpYVW.R;UTMJWFuxsF'KemhXjgKY!.VTh--JVTQ e,\nmNMLB UVFpcRtzUYz"` |
| 训练后样例 | `"\nNot meting mine, of your bare achiard and\nTo brother's roy. My elsey deceets: what is hope as inteed\nHas of when is away!\nSay'st Rown, thin never tranous Glory Romizage,\nStand a taunters thou sped usuch to Romal,\nEndly would pitilusty, for a should e"` |

## 问题与结论

- **遇到的问题与定位**：
  1. *网络连接偶发卡顿*：国内超算偶尔连接 GitHub 较慢，通过终端断点重试与标准 Git 流程保证两端同步。
  2. *路径丢失*：终端重启后默认进入系统根目录 `/#`，通过显式 `cd /root/private_data/Hands-on-LLM-Training` 切回持久化目录。
  3. *生成文本有拼写与逻辑错乱*：字符级（65 词表）模型逐字母预测，且语料仅 1MB 小样本；但已能自主拼出大量真实英文词汇、古英语特色虚词（thou, 'tis）及标准台词排版，完全达到预期。
- **checkpoint 重新加载是否成功**：通过独立脚本 [generate.py](generate.py) 成功脱机读取 `minigpt_final.pt`，进入 eval 模式顺利生成不同 prompt（`\n`, `KING:`, `To be, or not to be`）文本，状态恢复机制完备。
- **本周结论与下一步**：
  - *结论*：第 1 周目标圆满达成！完整走通了“文本 → Tokenizer → Transformer → Loss → 反向传播 → 保存模型 → 独立生成”全链条闭环。模型参数量 10.75M，训练 Loss 稳步从 4.27 收敛至 1.45，超算 BW 卡吞吐达到 18.8 万 tokens/s。
  - *下一步*：开启第 2 周，升级至 BPE 子词 Tokenizer，探索更高吞吐、更大规模（~100M）的训练工程与学习率调度。


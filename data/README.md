# 数据说明

仓库只记录数据的来源、版本、使用条件和处理方式。原始数据、处理后的大文件保存在训练服务器的持久目录，不提交到 Git。

## 计划使用的数据

| 周次 | 数据 | 用途 | 状态 |
| --- | --- | --- | --- |
| 第 1 周 | [Tiny Shakespeare 原始文本](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt) | 跑通字符级 Mini GPT 训练 | 待获取 |
| 第 2 周 | [TinyStories 数据集](https://huggingface.co/datasets/roneneldan/TinyStories)的固定子集 | 100M 级训练工程实验 | 待确定子集 |

实际使用前，请在对应周的实验记录中补充：下载日期、原始文件版本或校验值、样本数、训练/验证划分、清洗规则、许可与使用限制。对比实验必须沿用相同的数据版本和划分。

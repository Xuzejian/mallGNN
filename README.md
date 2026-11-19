# mallGNN：履约违规店铺预测

一个可以直接在 CPU 上跑通的初版：**店铺基础特征 + 历史履约统计 + 店铺关联关系 → 未来 7 天履约违规风险分**。

提供多关系 GraphSAGE、只用店铺特征的 MLP 基线、模拟训练样本、按时间划分的训练/验证/测试、模型保存与加载、无需标签的批量预测。使用原生 PyTorch 实现，不依赖 PyG、DGL 或图数据库。

## 1. 直接运行

Python 3.10+，推荐使用独立虚拟环境。Linux/macOS：

```bash
git clone https://github.com/Xuzejian/mallGNN.git
cd mallGNN
git switch codex/fulfillment-risk-v1
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# CPU 版本安装更轻便；已有适用的 PyTorch 可以跳过这一行。
python -m pip install 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python -m mallgnn demo
```

`demo` 默认生成 1,000 家虚构店铺的三个时间快照，然后分别训练 MLP、GraphSAGE，最后从保存的 GraphSAGE 模型加载并预测。CPU 即可运行；运行时会输出 loss 和验证集 PR-AUC。

输出在 `artifacts/demo/`：

| 文件 | 内容 |
| --- | --- |
| `data/{train,valid,test}/` | CSV 样本和快照元信息 |
| `mlp/model.pt`、`graphsage/model.pt` | 权重、训练特征编码器、阈值、图配置 |
| `graphsage/metrics.json` | 三个时间快照的指标 |
| `graphsage/run.json`、`history.json` | 参数、时间范围、图规模、训练曲线记录 |
| `graphsage/feature_names.json` | 模型实际使用的特征名 |
| `graphsage/{train,valid,test}_predictions.csv` | 带标签的评估分数，未标注节点的 label 为 -1 |
| `comparison.json` | 两个模型在测试集的对比 |
| `risk_scores.csv` | 加载模型后输出的无标签预测结果 |

小规模快速检查：

```bash
python -m mallgnn demo --num-shops 100 --epochs 5 --work-dir artifacts/quick
```

同一输出目录不自动覆盖。再次运行时换一个 `--work-dir`。

## 2. 已附简单样本，也可以单独执行每一步

`examples/toy_data/` 已提交 40 家虚构店铺、三个时间快照的 CSV，不需要先运行生成器。这批小样本只用于检查格式和流程。

```bash
python -m mallgnn train \
  --train-dir examples/toy_data/train \
  --valid-dir examples/toy_data/valid \
  --test-dir examples/toy_data/test \
  --output-dir artifacts/toy-model \
  --epochs 10

python -m mallgnn predict \
  --data-dir examples/toy_data/test \
  --checkpoint-path artifacts/toy-model/model.pt \
  --output-path artifacts/toy-scores.csv
```

生成更大的数据：

```bash
python -m mallgnn generate --output-dir data/my_demo --num-shops 2000 --seed 42
python -m mallgnn train \
  --train-dir data/my_demo/train \
  --valid-dir data/my_demo/valid \
  --test-dir data/my_demo/test \
  --output-dir artifacts/gnn \
  --model-type graphsage
```

将 `--model-type` 改为 `mlp`，并更换输出目录，即可做无图基线。所有命令支持 `--help`。默认两层、隐藏维度 48、dropout 0.15、AdamW、最多 120 轮、验证集 PR-AUC 连续 20 轮不提升则早停。可通过 `--hidden-dim`、`--num-layers`、`--lr`、`--max-neighbors`、`--patience` 等调整。`--device cuda` 可使用已有的 CUDA 环境；本次验收使用 CPU。

## 3. 任务和时间定义

一个节点是一家店铺；一个样本是 `(shop_id, as_of)`，其中 `as_of=T` 是预测截止时点。

- 特征只能使用截止 T 已经可获得的基础属性和历史统计；7/30 天统计对应 `(T-7d,T]`、`(T-30d,T]`。
- 关系只使用 `observed_at <= T` 的记录；未来关系会过滤并记录数量。
- `label=1`：店铺在 **`(T,T+7d]` 内至少发生一次经确认的履约违规**；否则为 0。这是本初版采用的任务定义，接入时应统一超时发货、虚假发货、履约取消等具体口径。
- 只有标签观察窗口完整、判定已成熟的店铺才能标为 0；不能把未知结果或标签缺失直接当负样本。未标注店铺直接不写入 `labels.csv`，仍可作为图节点参与消息传递。
- `label_available_at` 是标签最终可用时间。代码要求训练标签在验证快照前可用，验证标签在测试快照前可用。

模拟数据按以下时间拆分，**不做随机节点拆分**：

| 集合 | 特征截止 T | 未来标签窗口 | 标签可用时间 |
| --- | --- | --- | --- |
| train | 2026-01-31 | `(01-31,02-07]` | 2026-02-09 |
| valid | 2026-02-28 | `(02-28,03-07]` | 2026-03-09 |
| test | 2026-03-31 | `(03-31,04-07]` | 2026-04-09 |

同一家店可出现在不同时点，这是对存量店铺未来风险的预测。模型没有可训练的 shop_id embedding，新店也能用自身属性和当前已知关系打分。特征编码器只在训练快照上拟合；模型轮次按验证集 AP 选择，分类阈值按验证集最大 F1 选择；测试集不参与这两步选择。阈值需要等待验证标签成熟，因此预测接口还会检查模型可用时间。

所有时间统一解析为 UTC；日期字符串表示 UTC 当日 00:00。真实数据若采用北京时间自然日，建议传入带 `+08:00` 的完整时间戳，并对齐统计边界。`feature_time`、`observed_at` 必须代表平台实际已知时间；仅用业务发生时间无法阻止迟到数据或回填数据泄漏。

## 4. 真实数据接口

每个快照目录包含以下文件，具体列定义见 [docs/data_contract.md](docs/data_contract.md)：

```text
snapshot/
  metadata.json
  shops.csv
  stats.csv
  relations.csv
  labels.csv       # 训练/验证/测试需要；预测不需要，也不会读取
```

预测输出按 `risk_score` 从高到低排序，包含 `shop_id`、`as_of`、`risk_score`、`prediction` 和 `threshold`。`prediction` 采用 `risk_score >= threshold`。由于使用了类别加权损失，`risk_score` 应作为排序风险分，尚未进行概率校准。

统计数据允许个别数值缺失；使用训练中位数填充，并加入缺失指示。无统计的新店可以完全省略 `stats.csv` 中对应行；零订单的比率视为缺失。类目、地区以训练词表做 one-hot，未见取值进入 UNKNOWN；数值做适当 `log1p`、标准化和截断。ID、时间戳和未来标签均不进入模型特征。

## 5. 模型如何利用关联关系

| 关系类型 | 含义 | 使用建议 |
| --- | --- | --- |
| `same_owner` | 同一经营主体 | 第一版强关联候选 |
| `same_device` | 共用登录/经营设备 | 结合设备可信度、共用频次构建边权 |
| `same_warehouse` | 共用发货仓库 | 共享公共仓可能噪声很大，应限制规模 |

先用线性层编码店铺特征，再经过两层关系感知 GraphSAGE：对每种关系分别做加权邻居均值聚合，使用各自的可训练变换，再将当前节点存在的各关系消息求均值，并与自身变换相加。最后拼接原始店铺编码与图编码，输出风险分。标签不作为邻居输入。

关系按无向关系处理并扩为双向；自环去除、重复/反向重复取最大权重。每节点每种关系最多保留 30 个入邻居（可配置），优先保留高权重，权重相同时用稳定哈希决定，避免依赖 CSV 行顺序。截断后不保证双向对称。孤立店铺仍通过自身特征打分。

边权决定**同一种关系内部**邻居的相对贡献，不会因为把某一种关系的所有权重都乘 0.1 就降低该关系整体贡献；关系贡献通过可训练矩阵学习。自定义关系可通过 `--relation-types same_owner same_device same_ip` 配置，相应修改 CSV 即可；名称须为小写字母开头的字母、数字、下划线组合。

图计算使用稀疏边表，无 N×N 邻接矩阵；当前仍是全图训练，适合先做中小规模验证。大规模店铺图需要进一步接入邻居采样、小批量训练和离线图存储。上游建边时就应处理公共仓/公共 IP 超级节点，避免先生成巨大的全连接边表再截断。

## 6. 指标和复现

记录 ROC-AUC、PR-AUC（这里使用 Average Precision / AP）、precision、recall、F1，以及风险分最高 10% 店铺的命中率和召回率。业务上线前，应根据审核容量、误报成本和召回目标重新选择阈值，不能直接沿用模拟数据阈值。

`examples/verification.json` 记录一次真实执行的 1,000 家模拟店铺测试结果与环境版本。模拟器先生成潜在经营风险和历史行为，再独立采样未来标签；潜在风险不会输出为特征。**模拟数据结果仅用于验证代码与图信号能否跑通，不能作为线上效果或图模型必然优于基线的依据。**

```bash
python -m unittest discover -s tests -v
```

测试覆盖加权聚合、图节点重排、关系梯度、未来数据隔离、标签成熟时间、图去重和邻居上限、未知类目与缺失统计、新店/孤立节点，以及训练后加载模型进行无标签预测。GitHub Actions 还会运行 CLI demo。

## 7. 代码位置与后续开发

| 代码 | 职责 |
| --- | --- |
| `src/mallgnn/schema.py` | 特征白名单和默认关系 |
| `src/mallgnn/data.py` | 数据读取、标签/时间校验 |
| `src/mallgnn/features.py` | 比率/趋势衍生、缺失处理、编码 |
| `src/mallgnn/graph.py` | 建图、去重、边权、邻居限制 |
| `src/mallgnn/model.py` | 多关系 GraphSAGE / MLP |
| `src/mallgnn/pipeline.py` | 训练、评估、持久化、预测 |
| `src/mallgnn/synthetic.py` | 可复现的虚构样本生成 |
| `src/mallgnn/cli.py` | 命令行和一键 demo |

初版每个集合使用一个时间快照。接入真实业务后，建议先统一标签及负样本成熟口径，按多个月份滚动回测；按违规类型、店龄、类目和关联强度分层评估，并增加 LightGBM 等表格基线及关系消融。后续再扩展多快照联合训练、邻居采样、分数校准和解释能力。

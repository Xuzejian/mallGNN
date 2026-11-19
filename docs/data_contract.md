# 数据契约 v1

CSV 必须有表头、使用 UTF-8。`shop_id` 按字符串读取，可以包含前导零，每个快照内唯一。输入目录不要包含多个时间版本的同一店铺；一个目录就是一张时点图。额外业务列不会自动成为特征，要新增特征需修改 `schema.py` 和对应衍生逻辑。

## metadata.json

```json
{
  "schema_version": 1,
  "as_of": "2026-03-31T00:00:00+08:00",
  "label_horizon_days": 7
}
```

`as_of` 是所有节点共同的预测截止时点。训练、验证、测试必须采用相同预测窗口长度。

## shops.csv：基础特征

| 列 | 类型 | 说明 |
| --- | --- | --- |
| shop_id | string | 店铺 ID，非模型特征 |
| feature_time | timestamp | 此行基础特征已知时间，必须 ≤ as_of |
| shop_age_days | number | 截止 as_of 的店龄，天 |
| category | string | 主营类目，类别编码 |
| province | string | 经营地区，类别编码 |
| merchant_level | number | 商家等级，初版按有序数值处理 |
| deposit_amount | number | 保证金金额；所有金额字段使用统一币种 |
| is_enterprise | 0/1 | 是否企业店 |

数值特征非负；个别缺失值可留空，由训练中位数填充。等级字段的业务语义需保持一致。

## stats.csv：历史统计特征

| 列 | 类型 | 说明 |
| --- | --- | --- |
| shop_id | string | 对应 shops.csv，可只提供部分店铺 |
| feature_time | timestamp | 此行统计实际可用时间，必须 ≤ as_of |
| orders_7d | number | 近 7 天订单数 |
| orders_30d | number | 近 30 天订单数 |
| gmv_30d | number | 近 30 天订单总金额 |
| late_dispatch_orders_7d | number | 近 7 天延迟发货订单数 |
| late_dispatch_orders_30d | number | 近 30 天延迟发货订单数 |
| cancel_orders_30d | number | 近 30 天履约取消订单数 |
| refund_orders_30d | number | 近 30 天退款订单数 |
| complaint_orders_30d | number | 近 30 天被投诉的订单数，按订单去重 |
| historical_violation_orders_30d | number | 近 30 天已确认违规的订单数，仅使用截止 T 已知记录 |
| avg_dispatch_hours_30d | number | 近 30 天平均发货耗时，小时 |

各订单子集数量不能超过对应窗口订单数；orders_7d ≤ orders_30d、7 天延迟数 ≤ 30 天延迟数。不要直接填投诉事件次数等可能大于订单数的统计。整张表可以没有数据行，但仍需保留表头；适用于所有店铺均没有历史统计的极端情况。

内部衍生：7/30 天延迟率、30 天取消率/退款率/投诉率/历史违规率，7 天日均单量 ÷ 30 天日均单量，30 天客单价，是否存在统计。零订单的比率是缺失，不能被视为“零风险”。

这里只定义了接入聚合结果的接口，尚未实现订单明细 SQL/ETL。上游须确保所有统计的订单归属、去重方式、统计边界和数据可见时间一致。

## relations.csv：店铺关系

| 列 | 类型 | 说明 |
| --- | --- | --- |
| src_shop_id | string | 一端店铺 ID |
| dst_shop_id | string | 另一端店铺 ID |
| relation_type | string | 默认 same_owner / same_device / same_warehouse |
| weight | number | 正有限数，关系内部的相对可信度/强度 |
| observed_at | timestamp | 平台知道此关系的时间 |

每条关系写一个方向即可，构图时自动添加反向边。无任何关系时提供只有表头的 CSV。当前时点有效关系的端点必须在 shops.csv 内；未知关系类型、端点、非法边权会报错。未来关系记录会被过滤。初版关系一旦观测便视为有效，不含失效时间；真实业务应在上游导出截止 T 仍有效的关系。

不能根据未来处罚、人工调查结果或未来共享设备记录倒推历史关系。示例仅用虚构 shop_id，不包含真实主体、设备或仓库标识。

## labels.csv：未来结果

| 列 | 类型 | 说明 |
| --- | --- | --- |
| shop_id | string | 已成熟标签对应的店铺 |
| label | 0/1 | 未来窗口内是否至少发生一次履约违规 |
| label_start | timestamp | 必须等于 as_of，左开区间 |
| label_end | timestamp | 必须等于 as_of + label_horizon_days，右闭区间 |
| label_available_at | timestamp | 标签完整可用时间，不早于 label_end |

每家店最多一行。未标注店铺省略该行，不能填写 -1；内部才会以 -1 标记并从监督损失、评估中排除。训练、验证、测试每份数据都需要已标注的正负样本。预测时不读取该文件，不要求提供未来结果。

允许同一店铺在不同快照重复出现；若业务要评价新商家泛化，应另行构造跨店铺隔离的时间测试集，并关注图中关联主体重叠。

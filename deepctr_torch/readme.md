2023/02/06
我的一些关于DeepCTR的理解。

# 一些变量说明
- sparse_features、dense_features: 要使用到的特征中，稀疏特征和稠密特征说明。
- fixlen_feature_columns： 给要使用到的特征进行一层封装，给稀疏特征加上Embedding_dim。
  - linear_feature_columns: 宽深网中，宽侧的特征
  - dnn_feature_columns: 宽深网中，深侧的特征


# 模型训练

```python
model.compile()
model.fit()
```


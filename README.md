## 目標
本專案的目標是使用PyTorch框架，在CIFAR-10資料集上構建並訓練一個兩層神經網路，以解決圖像分類問題。專案的重點在於探索不同的超參數設置：權重初始化方法、優化器選擇以及資料增強技術對模型性能的影響，旨在提高分類準確率。
## 定義問題
  - 分類問題
  - X is a real number and has 3x32x32=3072 values(input nodes).
  - f(x) is a real number and has 10 values(output nodes).
  - Y is a real number and has 1 value.
## 實驗方法1 - 最佳的超參數設定
- **網絡架構：**
  - Input nodes：3x32x2=3072（對應CIFAR-10圖片展平後的維度）
  - Hidden nodes：1000
  - Output nodes：10（對應CIFAR-10的類別數）

- **數據預處理與增強：**
  - 隨機水平翻轉：`transforms.RandomHorizontalFlip()`
  - 隨機裁剪：`transforms.RandomCrop(32, padding=4)`
  - 標準化：`transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

- **權重初始化：**
  - Kaiming 均勻初始化：使用`nn.init.kaiming_uniform_`

- **激活函數：**
  - ReLU函數：使用`torch.nn.ReLU`

- **優化器：**
  - 優化器類型：Adam
  - 學習率：0.001
  - L2正則化權重衰減（λ）：0.001

- **學習率調度器：**
  - 余弦退火調度器：`torch.optim.lr_scheduler.CosineAnnealingLR`
  - T_max（周期）：設為與訓練的epoch數相同
  - eta_min（最小學習率）：0


## 實驗小結1
第五次實驗加上Cosine annealing decay schedlue後測試集上的準確率達52.94%，數據增強、增加隱藏層節點數、Kaiming搭配ReLU激活函数、Adam optimizer可以再增加近2%。
- 隱藏層節點數並非越多越好，需要尋找平衡點。
- Kaiming初始化對於ReLU激活函數至關重要，有助於減輕梯度消失或爆炸的問題。
- 優化器的選擇對模型性能有顯著影響。在本項目中，當學習率設置不當（例如過大的學習率0.01）時，Adam優化器的性能可能不如帶動量的SGD。
- 學習率調度對模型收斂和最終性能有積極作用，餘弦退火學習率調度在本實驗中提升了模型的準確率。

## 實驗方法與小結2 - Stop Criteria比較
![image](https://github.com/AngelaHsu02/Algorithm-2-Layer-Neural-Network/assets/128824007/f8e23e6a-355d-4a20-8142-5371c1d53f6f)
- 比較stopping criteria: weight tuning Epoch Bound(EB), Learning Goal(LG), EBorLG，設定EB = 10, LG = epsilon < 1.50，LG, EBorLG迭代到第8次，因為minibatch average train loss < epsilon停止調整權重。
- 可做調整: 降低epsilon，因為EB設計是因為當LG無法收斂到假設epsilon，才使用EB做停止標準，所以LG, EBorLG降低epsilon有機會accuracy更高。

![image](https://github.com/AngelaHsu02/Algorithm-2-Layer-Neural-Network/assets/128824007/93956222-56c3-43ac-9386-d0064e5af30a)
- 比較stopping criteria: EB, LG, EBorLG, LGorUA, EBorLGorUA
- 可發現**有UA**vs**無UA**分群效果：UA作為停止條件者，因為weight adjustment使loss下降較快。




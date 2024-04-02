## 目標
本專案的目標是使用PyTorch框架，在CIFAR-10資料集上構建並訓練一個兩層神經網路，以解決圖像分類問題。專案的重點在於探索不同的超參數設置：權重初始化方法、優化器選擇以及資料增強技術對模型性能的影響，旨在提高分類準確率。
## 方法

![image](https://github.com/AngelaHsu02/Algorithm-2-Layer-Neural-Network/assets/128824007/faedb0d0-5f81-4a70-b79a-9b97c69a881a)
![image](https://github.com/AngelaHsu02/Algorithm-2-Layer-Neural-Network/assets/128824007/bb9ed0bc-b7a2-4ba0-b6a6-53e176aeaff8)

第八次實驗Accuracy最佳54.75%，使用的超參數設定：
- **數據預處理與增強：**
  - 隨機水平翻轉：`transforms.RandomHorizontalFlip()`
  - 隨機裁剪：`transforms.RandomCrop(32, padding=4)`
  - 標準化：`transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))`

- **網絡架構：**
  - 輸入層維度：3072（對應CIFAR-10圖片展平後的維度）
  - 隱藏層節點數：1000
  - 輸出層維度：10（對應CIFAR-10的類別數）

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


## 結論
第五次實驗加上Cosine annealing decay schedlue後測試集上的準確率達52.94%，數據增強、增加隱藏層節點數、Kaiming搭配ReLU激活函数、Adam optimizer可以再增加近2%。
- 隱藏層節點數並非越多越好，需要尋找平衡點。
- Kaiming初始化對於ReLU激活函數至關重要，有助於減輕梯度消失或爆炸的問題。
- 優化器的選擇對模型性能有顯著影響。在本項目中，當學習率設置不當（例如過大的學習率0.01）時，Adam優化器的性能可能不如帶動量的SGD。
- 學習率調度對模型收斂和最終性能有積極作用，餘弦退火學習率調度在本實驗中提升了模型的準確率。
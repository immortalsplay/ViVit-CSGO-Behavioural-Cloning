import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import matplotlib.pyplot as plt
from tqdm import tqdm

from csgo_dataset import CSGODataset,SmallCSGODataset
from temporal import CSGO_model
from config import Config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建数据集
cs_go_dataset = CSGODataset(Config["file_path"]) # 读取全部数据
small_cs_go_dataset = SmallCSGODataset(Config["file_path"], num_samples=7,frames_per_sample=Config["horizon"]) # 读取前100个数据

# 创建 DataLoader
# dataloader = DataLoader(cs_go_dataset, batch_size=Config[batch_size], shuffle=True) # 读取全部数据
small_data_loader = DataLoader(small_cs_go_dataset, batch_size=Config["batch_size"], shuffle=True) # 读取前100个数据

# 初始化模型
model = CSGO_model(horizon = Config["horizon"], 
                   num_feature=Config["num_feature"],
                   depth = Config["depth"],
                   num_heads = Config["num_heads"],
                   head_dim = Config["head_dim"],
                   inverse_dynamic_dim = Config["inverse_dynamic_dim"]
                   )

# 将模型移到设备上
model.to(device)

# 设置损失函数和优化器
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# 开始训练
model.train()
epoch = Config["num_epochs"]
losses = []  # 用于绘制损失曲线
for epoch in range(epoch):
    epoch_loss = 0
    pbar = tqdm(enumerate(small_data_loader), total=len(small_data_loader), desc=f"Epoch {epoch + 1}/{epoch}")  # 进度条 替换成small_data_loader
    for batch_idx, (data, label) in pbar:
        # print(f"Data shape: {data.shape}, Label shape: {label.shape}")
        data = data.to(device)
        label = label.to(device)
        optimizer.zero_grad()

        outputs = model(data)

        # 计算损失
        loss = 0
        for i in range(26):
            label = label.float()
            loss += criterion(outputs.squeeze(2), label[:, 0, :].squeeze(1))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(small_data_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")
    losses.append(avg_loss)
    scheduler.step(avg_loss)  # 更新学习率

# 绘制损失曲线
plt.plot(range(1, epoch + 1), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# 保存模型
torch.save(model.state_dict(), "csgo_model.pth")
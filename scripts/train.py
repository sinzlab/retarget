import torch
import torch.nn as nn
from retarget.model import TransformerAutoEncoder

d_model = 512
nhead = 4
num_layers = 3
model = TransformerAutoEncoder(d_model=d_model, nhead=nhead, num_layers=num_layers)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

pose = animation.positions
trajectory = pose[:, 0]
pose = pose - trajectory[:, None]
pose = torch.Tensor(pose[:128][::16])
t_pose = torch.Tensor(animation.t_pose).unsqueeze(0)
edges = torch.LongTensor(animation.edges.T)

print(pose.shape, t_pose.shape, edges.shape)

num_epochs = 2000
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output, embedding = model(pose, t_pose, edges)
    loss = criterion(output, pose)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

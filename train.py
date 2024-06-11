import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import net  # 假设 Net 类定义在 net.py 中
from torch.utils.data import DataLoader, Dataset
import os
from sample import InfiniteSamplerWrapper


class MyDataset(Dataset):
    def __init__(self, content_dir, style_dir, transform=None):
        self.content_dir = content_dir
        self.style_dir = style_dir
        self.content_files = os.listdir(content_dir)
        self.style_files = os.listdir(style_dir)
        self.transform = transform

    def __len__(self):
        return min(len(self.content_files), len(self.style_files))

    def __getitem__(self, idx):
        content_img_name = os.path.join(self.content_dir, self.content_files[idx])
        style_img_name = os.path.join(self.style_dir, self.style_files[idx])
        content_image = Image.open(content_img_name).convert('RGB')
        style_image = Image.open(style_img_name).convert('RGB')

        if self.transform:
            content_image = self.transform(content_image)
            style_image = self.transform(style_image)

        return content_image, style_image


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        # 随机裁剪图像为256x256像素
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)


def adjust_learning_rate(optimizer, iteration_count, lr=1e-4):
    new_lr = lr / (1.0 + 5e-5 * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


# 定义模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_pth = 'models/vgg_normalised.pth'
decoder_pth = 'models/decoder.pth'
decoder = net.decoder
vgg = net.vgg

# 加载预训练模型参数
decoder.eval()
vgg.eval()
decoder.load_state_dict(torch.load(decoder_pth))
vgg.load_state_dict(torch.load(vgg_pth))
vgg = nn.Sequential(*list(vgg.children())[:31])
vgg.to(device)
decoder.to(device)

# 创建网络对象
model = net.Net(vgg, decoder)
model.train()
model.to(device)

# 加载图像
content_dir = './input/content'
style_dir = './input/style'
dataset = MyDataset(content_dir, style_dir, transform=train_transform())
data_iter = iter(DataLoader(
    dataset, batch_size=8,
    sampler=InfiniteSamplerWrapper(dataset)
))

# 定义优化器
optimizer = optim.Adam(model.decoder.parameters(), lr=1e-4)

# 定义训练过程
content_losses = []
style_losses = []
alpha = 1.0
max_iter = 20

for i in range(max_iter):
    adjust_learning_rate(optimizer, i)
    content_images, style_images = next(data_iter)
    content_images = content_images.to(device)
    style_images = style_images.to(device)
    loss_c, loss_s = model(content_images, style_images, alpha=alpha)
    loss = loss_c + loss_s

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    content_losses.append(loss_c.item())
    style_losses.append(loss_s.item())
    print(f'Iter: {i + 1}, Content Loss: {loss_c.item():.4f}, Style Loss: {loss_s.item():.4f}')
    if (i + 1) == max_iter:
        state_dict = model.decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].to(torch.device('cpu'))
        torch.save(state_dict, f'models/trained_decoder.pth')

print('Training completed!')
plt.figure()
plt.plot(content_losses, label='Content Loss')
plt.plot(style_losses, label='Style Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss During Training')
plt.show()

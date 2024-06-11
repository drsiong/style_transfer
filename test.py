import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(__file__))
import net
from function import adaptive_instance_normalization


def test_transform(size, crop):
    transform_list = []
    # if size != 0:
    #     transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    # TODO
    transform = transforms.Compose(transform_list)
    return transform

# 返回T（c，s）
def style_transfer(vgg, decoder, content, style, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    content_f = vgg(content)
    style_f = vgg(style)

    feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)

# 读取图片并调整大小
def resize_image(image_path, size):
    image = Image.open(image_path)
    image = image.resize(size, Image.Resampling.LANCZOS)
    return image

def main(content_pth='./input/content/1.jpg',
         style_pth='./input/style/1.jpg',
         content_size=512, style_size=512,
         alpha=1.0, crop=False, device='cpu'):
    save_ext = '.jpg'
    output_dir = 'output'
    vgg_pth = 'models/vgg_normalised.pth'
    decoder_pth = 'models/trained_decoder.pth'

    decoder = net.decoder
    vgg = net.vgg
    decoder.eval()
    vgg.eval()
    decoder.load_state_dict(torch.load(decoder_pth))
    vgg.load_state_dict(torch.load(vgg_pth))
    vgg = nn.Sequential(*list(vgg.children())[:31])
    vgg.to(device)
    decoder.to(device)

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)

    content = content_tf(Image.open(content_pth))
    style = style_tf(Image.open(style_pth))

    style = style.to(device).unsqueeze(0)
    content = content.to(device).unsqueeze(0)

    with torch.no_grad():
        output = style_transfer(vgg, decoder, content, style, alpha)
    output = output.cpu()

    output_name = f'{output_dir}/result{save_ext}'
    save_image(output, output_name)





if __name__ == '__main__':
    # vgg_pth = 'models/vgg_normalised.pth'
    # decoder_pth = 'models/decoder.pth'
    # decoder = net.decoder
    # vgg = net.vgg
    # decoder.eval()
    # vgg.eval()
    # decoder.load_state_dict(torch.load(decoder_pth))
    # vgg.load_state_dict(torch.load(vgg_pth))
    # vgg = nn.Sequential(*list(vgg.children())[:31])
    # model = AdaIN(vgg, decoder)
    # model.eval()

    content_pth = './input/content/avril.jpg'
    style_pth = './input/style/asheville.jpg'
    output_pth = './output/result.jpg'
    main(content_pth, style_pth)

    # 读取图片
    # 设定统一大小
    size = (300, 300)

    style_img = resize_image(style_pth, size)
    content_img = resize_image(content_pth, size)
    output_img = resize_image(output_pth, size)
    # 创建一个图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示风格图
    axes[0].imshow(style_img)
    axes[0].set_title('Style Image')
    axes[0].axis('off')  # 不显示坐标轴

    # 显示内容图
    axes[1].imshow(content_img)
    axes[1].set_title('Content Image')
    axes[1].axis('off')  # 不显示坐标轴

    # 显示融合图
    axes[2].imshow(output_img)
    axes[2].set_title('Output Image')
    axes[2].axis('off')  # 不显示坐标轴

    # 展示图形
    plt.show()
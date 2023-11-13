import cv2
from net.unet_big import UNetBig
import time
from utils.transform import train_transform
from utils.resize_image_keep_scale import resize_image_fixed
import torch
import os
from torchvision.utils import save_image
from PIL import Image

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = UNetBig(in_channels=3, out_channels=1, WithActivateLast=False).to(device)
    weights = '/home/highler/racecarVision_ws/src/visionmsg/detect_vehicle_line/pth/2023_epoch_25_loss_7.140139868244688e-11.pth'
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights, map_location=device))
        print('successfully')
    else:
        print('no loading')
    with torch.no_grad():
        image_path = '/home/highler/racecarVision_ws/src/visionmsg/detect_vehicle_line/dataset/train/images/00000001.jpg'
        image = Image.open(image_path)
        start_time = time.time()
        img = resize_image_fixed(image)
        img_data = train_transform(img).to(device)
        print(img_data.shape)
        img_data = torch.unsqueeze(img_data, dim=0)
        out = net(img_data)
        img_show = out.cpu().numpy().squeeze(0).transpose((1, 2, 0))
        img_show = cv2.cvtColor(img_show, cv2.COLOR_RGB2BGR)
        end_time = time.time()
        cv2.imshow('img', img_show)
        cv2.waitKey(0)
        print('total time: ', end_time - start_time)
        save_image(out, './result/result.jpg')

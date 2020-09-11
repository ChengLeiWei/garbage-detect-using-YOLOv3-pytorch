import torch
from torchsummary import summary
from models import *


cfg = r"./cfg/yolov3-ecbam.cfg"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
darknet_rfb = Darknet(cfg=cfg, img_size=(768, 448)).to(device)

summary(darknet_rfb, (3, 768,448))
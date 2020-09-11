# coding = utf-8
import numpy as np
import torch
import torch.nn as nn
from utils import parse_config
from models import create_modules
from models import *
from pathlib import Path
from utils.utils import *
import os
from PIL import Image, ImageFont, ImageDraw
import cv2
import freetype
from utils.layers import *
from utils.torch_utils  import *


# print(cv2.__version__)
# ft2 = freetype.create_string_buffer()
# # ft2.loadFont

# TODO 问题一
# imgsz_min, imgsz_max = 300, 400
# multi_scale |= imgsz_min != imgsz_max
# if multi_scale:
#     print(multi_scale)

# TODO 问题二
# key = 'size'
# val = [',']
# if key == 'size' and ',' in val:
#     print(True)

# ONNX = False
# TODO 问题三
# class Mylinear(nn.Module):
#     def __init__(self, name, verbose=False):
#         super(Mylinear, self).__init__()
#         self.info(verbose) if not ONNX else None
#         self.name = name
#         self.hidden = nn.Sequential('linear', nn.Linear(3, 1))
#
#     def info(self):
#         print()


# class_1 = Myclass('c')
# print()

# # # TODO: 问题4 查看mdef和routs的形状
# path_1 = 'cfg/yolov3-haihua.cfg'
# mdef_1 = parse_config.parse_model_cfg(path_1)
# print("*" * 10 + "解析的模型字典" + "*" * 10)
# for i, layer_dict in enumerate(mdef_1):
#     print(i, end=' ')
#     print(layer_dict)

# print("*" * 10 + "creat_modules构建的模型列表" + "*" * 10)
# module_list, routs, out_filters = create_modules(mdef_1, 416)
# for id, module in enumerate(module_list):
#     print(id, ' ', module)
# #
# print("*" * 10 + "creat_modules构建的模型的routs" + "*" * 10)
# for j, val in enumerate(routs):
#     print(j, val)
#
# print("*" * 10 + "creat_modules构建的模型输出记录out_filters" + "*" * 10)
# for k, filter in enumerate(out_filters):
#     print(k, filter)

# TODO： 问题5 查看self.anchor_vec 形状
# anchor = np.array([[135, 164],
#                   [161, 190]], dtype=np.float)
# anchor_tensor = torch.tensor(anchor)
# anchor_tensor_grid = anchor_tensor / 13
# anchor_vec = anchor_tensor_grid.view(1, 2, 1, 1, 2)
# print(anchor_vec)

# TODO： 问题6 [[]] *n,是什么结果
# batchs_shape = [[1, 1]] * 10
# print(batchs_shape)

# TODO: 问题7 查看Darknet各层输出形状
# model = Darknet('./cfg/yolov3-haihua-rfb.cfg', img_size=[768, 448])
# print(model)
# x = torch.rand(16, 3, 768, 448)
# for name, blk in model.named_modules():
#     X = model(x)
#     print(type(X))
#     print(name, blk)
#     print('out shape:',  X)

# TODO 问题7.1  查看self.children()
# model = Darknet('./cfg/yolov3-haihua-ecbam-trans32.cfg', img_size=[768, 448])
# for a in model.children():
#     print(a)
# # for para in model.state_dict():
# #     print(model.state_dict())
# for a, para in zip(model.children(), model.parameters()):
#     print(a)
#     print("*"*20)
#     print(para.size())

# for b in list(model.children()):
#     print(b)
# TODO: 问题8 nn,AdapativeAvgPool2d(1)是什么结果：
# class Myavgmodel(nn.Module):
#     def __init__(self):
#         super(Myavgmodel, self).__init__()
#         self.conv = nn.Conv2d(in_channels=1024,
#                               out_channels=1024,
#                               kernel_size=3,
#                               padding=1,
#                               bias=False)
#         self.avg = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, x):
#         x = self.conv(x)
#         return self.avg(x)
#
#
# x = torch.rand([16, 1024, 13, 13], dtype=torch.float)
# mymodel = Myavgmodel()
# X = mymodel(x) * x
# print(X.size())

# TODO: 问题9 torch.mean(x, dim=1, keepdim = True)结果
# x = torch.rand((16, 1024, 13, 13),dtype=torch.float)
# x_mean = torch.mean(x, dim=1, keepdim=True)
# x_max, _ = torch.max(x, dim=1, keepdim=True)
# x = torch.cat((x_mean, x_max), 1)
# print(x.size())

# # TODO: 问题10 查看'./weight/yolov3.weights的数据
# weights_path = '.weights/yolov3.weights'

# TODO : 问题11 查看model的参数
# cfg = './cfg/yolov3-haihua-rfb.cfg'
# device = ('cuda' if torch.cuda.is_available() else 'cpu')
# model = Darknet(cfg=cfg).to(device)
#
# pg0, pg1, pg2 = [], [], []
# pg0_dict = {}
# for k, v in dict(model.named_parameters()).items():
#     if '.bias' in k:  #
#         pg2 += [v]
#     elif 'Conv2d.weight' in k:
#         pg1 += [v]
#     else:
#         pg0 += [v]
#         pg0_dict[k] = pg0
# print(pg0, '\n', pg1, '\n', pg2, '\n')
# print(pg0_dict)


# # TODO 问题12 查看字符串是否在另一个字符串中
# str1 = 'BatchNorm2d.bias'
# str2 = '.bias'
# print(str2 in str1)

# TODO 问题13 lambda函数
# for epochs in range(10):
#     lf = lambda x: (((1 + math.cos(
#         x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05
#     print(lf)

# def compute_even(a, b=0, **c):
#     c['0'] = 1
#     c['1'] = 2
#     if a > b:
#         if a % c['1'] == 0:
#             print('%d是偶数' %(a))
#         else:
#             a += 1
#             return a
#     else:
#         print('请输入大于0的数')
#
# num = compute_even(a=2)
# print(num)


# # TODO 问题14 np.concatenate函数将labels拼接
# labels2 = []
# labels_1 = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]], dtype=np.int32)
# labels2.append(labels_1)
# labels_2 = np.array([[2, 2, 2, 2, 2]], dtype=np.int32)
# labels2.append(labels_2)
# lable2 = np.concatenate(labels2, 0)
# print(len(labels2))

# TODO 15 mod
# print(np.mod(5, -3))

# TODO 16 np.interp一维线性差值

# TODO 17 除法和取余优先级
# a = 8 / 4 % 4
# print(a)  优先级一样

# TODO 问题18 torch.linespace()
# device = ('cuda' if torch.cuda.is_available() else 'cpu')
# a = torch.linspace(0.5, 0.95, 10).to(device)
# print(a)
# a = a[0].view(1)
# print(a)

# TODO 问题19 meshgrid
# nx, ny = 56, 32
# yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
# grid = torch.stack([yv, xv], 2)
# print(yv, xv)
# print(grid.size()) --> (32, 56, 2)


# TODO 问题 20 &=
# nc = 1
# multi_label = True
# multi_label &= nc > 1
# print(multi_label)

# TODO 问题21 inference输出长度
# inference_out = torch.rand((16, 507, 85), dtype=torch.float64)
# infience_len = len(inference_out)
# print(infience_len)
# for x, xi in enumerate(inference_out):
#     print(len(xi))

# TODO  问题22 nonzeros
# a = torch.rand((3, 7), dtype=torch.float64)
# print(a)
# conf_thres = 0.5
# a_filter = (a[:, 5:] > conf_thres)
# print(a_filter)
# a_non0 = a_filter.nonzero()
# print(a_non0)
# i, j = (a[:, 5:] > conf_thres).nonzero().t()
# print(i, j)

# TODO 问题23 c.view(-1, 1) * wh_max
# c = torch.tensor([3, 4, 56, 23], dtype=torch.long)
# wh_max = 4600
# c = c.view(-1, 1) * wh_max
# print(c)

# TODO 问题24 torch.zeros()第二个形参是什么
# niou=10
# pred= torch.rand([4, 6], dtype=torch.float)
# correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
# print(correct)

# TODO 问题25 x是形状(1, 3, 2, 2)的tensor, 查看torch.max(x, 1)[0]是什么形状
# x = torch.randn([2, 3, 2, 2], dtype=torch.float64)
# print(x)
# max_x = torch.max(x, 1)  # 必须加[0]，因为torch.max()返回的是张量的最大值和最大值在其求最大值维度上的索引[0]
# # 因此我们要活的如果不加会报错，同理[1]
# print(max_x)

# TODO obj.argsort()如果在对象中的值是全部相同的，会如何返回索引？
# list1 = [9, 9, 9, 9, 9]
# np_list1 = np.array(list1).astype(np.float32)
# i = np_list1.argsort()
# print(i)

# TODO concatenate(wh, 0).repeat(nr, axis=0)
# a = np.array([[1, 10, 10, 0.76, 0.47], [1, 10, 10, 0.32, 0.63]])
# b = np.array([[1, 10, 10, 0.484, 0.634], [1, 10, 10, 0.561, 0.61]])
# wh = np.concatenate((a[:, 3:5], b[:, 3:5]), axis=0).repeat(2, axis=0)
# print(wh.shape)
#
# # TODO np.random.uniform(5, 5, size=(wh.shape[0], 1))
# wh *= np.random.uniform(540, 960, size=(wh.shape[0], 1))
# wh = wh[(wh > 2.0).all(1)]
# print(wh)

# TODO 常识写一下LoadImagesAndLabels类
# img_ext = ['.jpg']
# class My_LoadImagesAndLabels:
#     # 需要 image_file_path:train image 的txt文件
#     def __init__(self, path, batch_size, augument=False, rect=False):
#         path = str(Path(path))  # 这里为什么用Path？还有这里有str
#         assert os.path.isfile(path), ''
#         with open(path, 'r') as f:
#             self.file_list = [x.replace("/", os.sep) for x in f.read().splitlines()
#                               if os.path.splitext(x)[-1].lower() in img_ext]  # 有lower()
#         # 思考：这里
#         self.img_number = len(self.file_list)
#         assert self.img_number > 0, '没有发现图片文件'
#         # TODO batch_index是否可以定义为属性
#         bi = np.floor(np.arange(self.img_number) / batch_size).astype(np.int)
#         bn = bi[-1] + 1


# TODO 对ground_truth的wh和anchor_wh做wh_iou,查看形状
# anchor_wh = torch.tensor([[2, 3], [3, 4], [4, 4], [6, 3]], dtype=torch.float32)
# target_wh = torch.tensor([[5, 3], [3, 3], [5, 6]], dtype=torch.float32)
# target_wh = target_wh[:, None]
# anchor_wh = anchor_wh[None]
# inter = torch.min(target_wh, anchor_wh).prod(2)
# iou = inter / (target_wh.prod(2) + anchor_wh.prod(2) - inter)
# print(inter, iou)
# TODO 注意torch.min(dim=0, dim=1, dim=2),分别表示从dim=0,dim=1,dim=2方向去看,理解三个方向求
#  min或者max有一个比较简单的方法：投影——利用投影可判断每个方向看过去获得的tensor的形状

# TODO max(1).[0]
# wh = torch.empty((3, 4), dtype=torch.float32)
# wh = wh.random_(0, 3)
# print(wh)
# wh_max = wh.max(1)
# print(wh_max)

# TODO kmeans anchor尺寸
# k = kmean_anchors(path="./GarbageData/garbage_train.txt", n=12, img_size=(768, 768), thr=0.225)
# print(k)

# TODO 查看整数除法
# num1 = 201
# num2 = 400
# gain = num1 / num2
# print(gain)

# TODO ImageFont.truetype()
# text = '这是一个中文字符串'
# sample2_path = './data/sample2'
# leaf, top = 430, 300
# # for root, dirs, files in os.walk(sample2_path):
# #     for file in files:
# #         img = cv2.imread(os.path.join(root, file), cv2.IMREAD_COLOR)
# #         img_name = os.path.join(root, file)[:-4]
# #         cv2.imwrite(img_name + '.jpg', img)
#
# for root, dirs, files in os.walk(sample2_path):
#     for i, file in enumerate(files):
#         img = cv2.imread(os.path.join(root, file), cv2.IMREAD_COLOR)
#         if isinstance(img, np.ndarray):
#             print("img是Opencv格式")
#             img = Image.fromarray(cv2.cvtColor(img, code=cv2.COLOR_BGR2RGB))
#         img_ori = Image.new('RGB', )
#         draw = ImageDraw.Draw(img)
#         fontstyle = ImageFont.truetype("/home/wcl/yolo_project/yolov3-master/tools/font/simsun.ttc",
#                                        30, encoding="utf-8")
#         fontstyle2 = ImageFont.truetype()
#         print(fontstyle.getsize(text))
#         draw.text((leaf, top), text, (0, 255, 0), font=fontstyle)
#         img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
#
#         del draw  # 删除画笔
#         img.close()

        # font = ImageFont.truetype('3.ttf', 50)　  # 使用自定义的字体，第二个参数表示字符大小
        # im = Image.new("RGB", (50, 50))　　　　　　  # 生成空白图像
        # draw = ImageDraw.Draw(im)　　　　　　　　  # 绘图句柄
        # x, y = (0, 0)　　　　　　　　　　　　　　　　　　  # 初始左上角的坐标
        # draw.text((x, y), '3', font=font)　　　　  # 绘图
        # offsetx, offsety = font.getoffset('3')　　  # 获得文字的offset位置
        # width, height = font.getsize('3')　　　　　  # 获得文件的大小
        # im = np.array(im)
        # cv2.rectangle(im, (offsetx + x, offsety + y), (offsetx + x + width, offsety + y + height), (255, 255, 255),
        #               1)  # 绘出矩形框
        # imshow(im)

        # img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # cv2.imshow('show_%d' % i, img)
        # if cv2.waitKey(100000) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()


        # def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
#     if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
#         img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     # 创建一个可以在给定图像上绘图的对象
#     draw = ImageDraw.Draw(img)
#     # 字体的格式
#     fontStyle = ImageFont.truetype(
#         "font/simsun.ttc", textSize, encoding="utf-8")
#     # 绘制文本
#     draw.text((left, top), text, textColor, font=fontStyle)
#     # 转换回OpenCV格式
#     return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)
#
#
# if __name__ == '__main__':
#     img = cv2ImgAddText(cv2.imread('img1.jpg'), "大家好，我是片天边的云彩", 10, 65, (0, 0 , 139), 20)
#     cv2.imshow('show', img)
#     if cv2.waitKey(100000) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()

# print(cv2.__version__)

# TODO 测试是哪里报关于libpng的iccp警告
# name_path = './GarbageData/garbage.names'
# names = load_classes(name_path)
# print(type(names))
# for name in names:
#     print(name)


# TODO 查看for循环里的作用域
# sum = 1
# list_1 = [2, 3]
# list_length = len(list_1)
# if list_length:
#     for i in range(list_length):
#         val = list_1[i]
#         sum += val
# print(sum)

# rfb = BasicRFB(1024, 512, stride=1, visual=1, scale=1)
# print(rfb.__class__.__name__)

# TODO 写一个从1到204的列表
# x = []
# for k in range(1, 205):
#     x.append(k)
# print(x)

# TODO 查看是否RFB模块的输出形状相似？
# class MyRFB(nn.Module):
#
#     def __init__(self, img_size = (768, 448)):
#         super(MyRFB, self).__init__()
#         in_planes = 3
#         out_planes = 256
#         self.conv1 = BasicConv(in_planes, out_planes, 3, 1, 1)
#         self.rfb = BasicRFB(out_planes, 512, stride=1, visual = 2, scale=1)
#         self.conv2 = BasicConv(512, 100, 3, 1, 1)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.rfb(x)
#         x = self.conv2(x)
#         return x
#
# x = torch.rand((2, 3, 768, 448))
# rfb_2conv = MyRFB()
# for i, m in rfb_2conv.modules():
#     X = rfb_2conv(x)

# TODO tensor和Tensor
# p = torch.tensor((8, 3, 14, 24, 209))
# print(p)
# s_array = np.ndarray((3, 4))
# print(s_array.shape)
# s = torch.Tensor(s_array)
# print(s.shape)

# TODO repeat
# a = torch.arange(3).view(-1, 1).repeat(1, 16)
# print(a)
# t = torch.rand((16, 6))
# print(t)
# t = t.repeat(3, 1)
# print(t)

# TODO utils中的计算就>model.iou正例部分j是是什么
# iou = torch.rand((3, 16), dtype=torch.float64)
# j = iou.view(-1) > 0.4
# print(j, type(j), j.shape)
# t = torch.rand((48, 6),dtype=torch.float64)
# print(t, t.shape)
# t = t[j]
# print(t, t.shape)

# TODO zeros_like()
# pi = torch.from_numpy(np.ndarray((8, 3, 14, 24, 85), dtype=np.float32))
# tp = torch.zeros_like(pi[..., 0])
# print(tp.shape)

# s = np.arange(1, 10)
# print(s)
# a = np.ndarray((3, 4)).view()

# a = int(3.5)
# b = 3 // 2
# c = 3 % 2
# d = 5 % 3
# e = 5 % -3
# print(a, b, c, d, e)

# TODO 初始化测试
# cfg = 'cfg/yolov3-haihua-rfb.cfg'
# model = Darknet(cfg=cfg, img_size=[768, 448])
# m_no_init, kaiming_init_success_cnt, Batch_init_cnt, active_init_cnt = {}, 0, 0, 0
# for i, m in enumerate(model.modules()):
#     if isinstance(m, nn.Conv2d):
#         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#         kaiming_init_success_cnt += 1
#         print('-'*20 + 'kaiming_init_success!' + '-'*20)
#     elif isinstance(m, nn.LeakyReLU) or isinstance(m, nn.ReLU) or isinstance(m, nn.ReLU6):
#         m.inplace = True
#         active_init_cnt += 1
#         print('+' * 20 + 'active_init_success!' + '+' * 20)
#     elif isinstance(m, nn.BatchNorm2d):
#         m.eps = 1e-4
#         m.momentum = 0.03
#         Batch_init_cnt += 1
#         print('@' * 20 + 'Batch_init_success!' + '@' * 20)
#     else:
#         m_no_init[i] = m.__class__.__name__
# print('kaiming_init success times: %d' % kaiming_init_success_cnt)
# print('batch_init success times: %d' % Batch_init_cnt)
# print('active_init success times: %d' % active_init_cnt)
# print(m_no_init)

# TODO 获取每次修改模型的名字
# cfg = 'cfg/yolov3-haihua-rfb.cfg'
# cfg_name = str(cfg[4:-4])
# print(cfg_name)

# TODO 查看 c |= a !=b  # Prior：比较运算符>位逻辑运算>逻辑运算
# a = 1
# b = 20
# c = False
# c |= a != b
# print(c)

# my_img = r'./xlpic.jpeg'
# img_jpg = cv2.imread(my_img)
# cv2.imwrite('my_photo.png', img_jpg)

# TODO 列表元素超出数值表示范围
# dir1 = {}
# dir1['filters'] = int(3072) if (int(3072) - float(3072))==0 else float(3072)
# print(dir1)

import utils.parse_config
cfg_path = r"./cfg/yolov3-haihua-assistant--35-rfbs-61-rfbs-distance-1-to-head.cfg"
mdef = parse_model_cfg(cfg_path)
print(mdef)

import os
import random
import shutil

from tools.check_jpgAndxml import checkJpgXml
from tools.create_main import create_main_txts
from tools.voc_label import gen_voc_lable
from tools.make_for_yolov3_torch import make_for_torch_yolov3


'''
    create_all文件主要解析VOC文件成为yolov3可用的txt形式
'''

if __name__ == "__main__":
    ###############################################################################
    # 0. 参数设置
    jpegimages_dir = r"./JPEGImages"  # 图片保存位置
    annotations_dir = r"./Annotations"  # 标注文件保存位置

    # 分为3份：train,val,test
    trainval_percent = 0.89  # (train+val)/(train+val+test)
    train_percent = 1  # (train)/(train+val)

    # 修改类别
    classes = ["PET", "HDPE", "PP", "glass", "cans", "paper", "Tetra pak packaging"]
    # 修改为自己的类别,多个类["class1","class2"]

    # dir_label = r"./voc_labels"
    dir_label = r"./trash_labels"

    # dir1_train = r"./images/train2007"
    # dir1_val = r"./images/test2007"
    # dir2_train = r"./labels/train2007"
    # dir2_val = r"./labels/test2007"
    dir1_train = r"./images/train2020"
    dir1_val = r"./images/test2020"
    dir2_train = r"./labels/train2020"
    dir2_val = r"./labels/test2020"

    main_trainval = r"./ImageSets/Main/trainval.txt"
    main_test = r"./ImageSets/Main/test.txt"
    
    ##################################################################################

    # 1. 检查jpg和xml文件是否是一一对应的
    print("=" * 5, "\t1. checking jpg and xml\t", "=" * 5)
    checkJpgXml(jpeg_dir=jpegimages_dir, annot_dir=annotations_dir)

    # 2. 按照比例创建训练、验证、测试
    print("=" * 5, "\t2. split train, val, test\t", "=" * 5)
    create_main_txts(trainval_percent=trainval_percent,
                     train_percent=train_percent)

    # 3. 将坐标进行归一化，生成labels中的txt文件
    print("=" * 5, "\t3. create txt files in labels\t", "=" * 5)
    gen_voc_lable(classes)

    # 4. 构建pytorch版本yolov3格式
    print("=" * 5, "\t4. create yolov3 format\t", "=" * 5)
    make_for_torch_yolov3(jpegimages_dir, dir_label, dir1_train, dir1_val,
                          dir2_train, dir2_val, main_trainval, main_test)

    # 5. 收尾，将中间文件删除
    print("=" * 5, "\t5. delete temporary directory\t", "=" * 5)
    if os.path.exists(dir_label):
        shutil.rmtree(dir_label)

    # 6. 之后的操作：
    print("="*50)
    print("配套代码: https://github.com/ultralytics/yolov3")
    print("之后的操作：")
    print("1. 将images,labels文件夹放到yolov3里的data文件夹")
    print("2. 将2007_train.txt 2007_test.txt文件放到yolov3里的data文件夹")
    print("3. 修改template.names, 里边就是classes内容，每行一个类")
    print("4. 修改template.data,修改类别个数classes")
    print("5. 修改yolov3/train.py中的coco.data为template.data")
    print("6. 严格按照以上步骤进行执行即可运行")
    print("="*50)
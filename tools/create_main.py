import os  
import random  


def create_main_txts(trainval_percent, train_percent):
    xmlfilepath = 'Annotations'  
    txtsavepath = 'ImageSets/Main'

    if not os.path.exists(txtsavepath):
        os.makedirs(txtsavepath)

    total_xml = os.listdir(xmlfilepath)  # listdir返回的是'文件名.xml'的标签文件名字符串
    num = len(total_xml)
    num_list = range(num)
    tv = int(num*trainval_percent)
    tr = int(tv*train_percent)
    trainval = random.sample(num_list, tv)
    train = random.sample(trainval, tr)
    
    ftrainval = open('ImageSets/Main/trainval.txt', 'w')  # 自动创建
    ftest = open('ImageSets/Main/test.txt', 'w')  
    ftrain = open('ImageSets/Main/train.txt', 'w')  
    fval = open('ImageSets/Main/val.txt', 'w')  
    
    for i in num_list:
        name = total_xml[i][:-4]+'\n'  # 由于total_xml存储的是xml文件的字符串，所以total_xml[i][:-4]
        # 中[:-4]就表示xml文件字符串从第一个名字字符开始到'.xml'扩展名之前的字符，也正是文件名字
        if i in trainval:  
            ftrainval.write(name)  
            if i in train:  
                ftrain.write(name)  
            else:  
                fval.write(name)  
        else:  
            ftest.write(name)  
    
    ftrainval.close()  
    ftrain.close()  
    fval.close()  
    ftest.close()


if __name__ == "__main__":
    pass
import os, shutil
from tqdm import *


def checkJpgXml(jpeg_dir, labels_dir):
    """
    dir1 是图片所在文件夹
    dir2 是标注文件所在文件夹
    """
    settype = 'val2014'
    nousefile_path = './cocodata/%s/%s_nouseImages' % (settype, settype)
    if not os.path.exists(nousefile_path):
        os.makedirs(nousefile_path)
    pBar = tqdm(total=len(os.listdir(jpeg_dir)))
    cnt = 0
    for file in os.listdir(jpeg_dir):
        pBar.update(1)
        f_name, f_ext = file.split(".")
        if not os.path.exists(os.path.join(labels_dir, f_name + ".txt")):
            print(f_name)
            cnt += 1
            old_file = os.path.join(jpeg_dir, file)
            nouse_file = os.path.join(nousefile_path, file)
            shutil.move(old_file, nouse_file)

    if cnt > 0:
        print("有%d个文件不符合要求。" % (cnt))
    else:
        print("所有图片和对应的xml文件都是一一对应的。")


dataType = 'val2014'
dir1 = r"./cocodata/%s/JPEGImages" % (dataType)
dir2 = r"./cocodata/%s/labels" % (dataType)
checkJpgXml(dir1, dir2)
# if __name__ == "__main__":
#     dir1 = r".\JPEGImages"
#     dir2 = r".\Annotations"
#     checkJpgXml(dir1, dir2)
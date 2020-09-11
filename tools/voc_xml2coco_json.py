# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 15:01:03 2018
需要改动xml_path and json_path
"""
# !/usr/bin/python
# -*- coding:utf-8 -*-
# @Description: xml转换到coco数据集json格式

import os, sys, json, xmltodict

from xml.etree.ElementTree import ElementTree, Element
from collections import OrderedDict

XML_PATH = "/home/learner/datasets/VOCdevkit2007/VOC2007/Annotations/test"
JSON_PATH = "./test.json"
json_obj = {}
images = []
annotations = []
categories = []
categories_list = []
annotation_id = 1


def read_xml(in_path):
    '''读取并解析xml文件'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree


def if_match(node, kv_map):
    '''判断某个节点是否包含所有传入参数属性
      node: 节点
      kv_map: 属性及属性值组成的map'''
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True


def get_node_by_keyvalue(nodelist, kv_map):
    '''根据属性及属性值定位符合的节点，返回节点
      nodelist: 节点列表
      kv_map: 匹配属性及属性值map'''
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes


def find_nodes(tree, path):
    '''查找某个路径匹配的所有节点
      tree: xml树
      path: 节点路径'''
    return tree.findall(path)


print("-----------------Start------------------")
xml_names = []
for xml in os.listdir(XML_PATH):
    # os.path.splitext(xml)
    # xml=xml.replace('Cow_','')
    xml_names.append(xml)

'''xml_path_list=os.listdir(XML_PATH)
os.path.split
xml_path_list.sort(key=len)'''
xml_names.sort(key=lambda x: int(x[:-4]))  # 这里的xml_names是带.jpg后缀的；
# 上面的x是xml_names
new_xml_names = []
for i in xml_names:
    j = 'Cow_' + i  # TODO 这里需要修改吧？？ 不需要加Cow
    new_xml_names.append(j)

# print xml_names
# print new_xml_names
for xml in new_xml_names:
    tree = read_xml(XML_PATH + "/" + xml)
    object_nodes = get_node_by_keyvalue(find_nodes(tree, "object"), {})  # TODO 这里的get_node_by_keyvalue的第二个参数送
    # 的{}，会造成什么影响
    if len(object_nodes) == 0:
        print(xml, "no object")
        continue
    else:
        image = OrderedDict()
        file_name = os.path.splitext(xml)[0]  # 文件名
        # para1 = file_name + ".jpg"
        para1 = file_name + ".png"
        height_nodes = get_node_by_keyvalue(find_nodes(tree, "size/height"), {})
        para2 = int(height_nodes[0].text)
        width_nodes = get_node_by_keyvalue(find_nodes(tree, "size/width"), {})
        para3 = int(width_nodes[0].text)

        fname = file_name[4:]
        para4 = int(fname)

        for f, i in [("file_name", para1), ("height", para2), ("width", para3), ("id", para4)]:
            image.setdefault(f, i)

            # print(image)
        images.append(image)  # 构建images

        name_nodes = get_node_by_keyvalue(find_nodes(tree, "object/name"), {})
        xmin_nodes = get_node_by_keyvalue(find_nodes(tree, "object/bndbox/xmin"), {})
        ymin_nodes = get_node_by_keyvalue(find_nodes(tree, "object/bndbox/ymin"), {})
        xmax_nodes = get_node_by_keyvalue(find_nodes(tree, "object/bndbox/xmax"), {})
        ymax_nodes = get_node_by_keyvalue(find_nodes(tree, "object/bndbox/ymax"), {})
        for index, node in enumerate(object_nodes):
            annotation = {}
            segmentation = []
            bbox = []
            seg_coordinate = []  # 坐标
            seg_coordinate.append(int(xmin_nodes[index].text))
            seg_coordinate.append(int(ymin_nodes[index].text))
            seg_coordinate.append(int(xmin_nodes[index].text))
            seg_coordinate.append(int(ymax_nodes[index].text))
            seg_coordinate.append(int(xmax_nodes[index].text))
            seg_coordinate.append(int(ymax_nodes[index].text))
            seg_coordinate.append(int(xmax_nodes[index].text))
            seg_coordinate.append(int(ymin_nodes[index].text))
            segmentation.append(seg_coordinate)
            width = int(xmax_nodes[index].text) - int(xmin_nodes[index].text)
            height = int(ymax_nodes[index].text) - int(ymin_nodes[index].text)
            area = width * height
            bbox.append(int(xmin_nodes[index].text))
            bbox.append(int(ymin_nodes[index].text))
            bbox.append(width)
            bbox.append(height)

            annotation["segmentation"] = segmentation
            annotation["area"] = area
            annotation["iscrowd"] = 0
            fname = file_name[4:]
            annotation["image_id"] = int(fname)
            annotation["bbox"] = bbox
            cate = name_nodes[index].text
            if cate == 'head':
                category_id = 1
            elif cate == 'eye':
                category_id = 2
            elif cate == 'nose':
                category_id = 3
            annotation["category_id"] = category_id
            annotation["id"] = annotation_id
            annotation_id += 1
            annotation["ignore"] = 0
            annotations.append(annotation)

            if category_id in categories_list:
                pass
            else:
                categories_list.append(category_id)
                categorie = {}
                categorie["supercategory"] = "none"
                categorie["id"] = category_id
                categorie["name"] = name_nodes[index].text
                categories.append(categorie)

json_obj["images"] = images
json_obj["type"] = "instances"
json_obj["annotations"] = annotations
json_obj["categories"] = categories

f = open(JSON_PATH, "w")
# json.dump(json_obj, f)
json_str = json.dumps(json_obj)
f.write(json_str)
print("------------------End-------------------")

"""
    Get table content from table content json files and add content to table structure json files.

    should input: polygon_path(location of table structure json files)
                  label_path(location of table content json files)
                  shutil_path(move table structure json files and images in this path if errors)

"""

import json
import os
import glob
from tqdm import tqdm
import shutil
from shapely.geometry import Point, Polygon


def add_label(label_file_path, polygon_file_path):
    with open(polygon_file_path, encoding='utf-8') as file:
        dic_polygon = json.load(file)
    with open(label_file_path, encoding='utf-8') as file:
        dic_label = json.load(file)

    k = len(dic_label["shapes"])  # 获取标签数目
    label_cell = []  # 存储标签信息的数组(中心横坐标，中心纵坐标，对应单元格内容序号)
    for i in range(0, k):  # 遍历每一个标签
        m = len(dic_label["shapes"][i]["points"])  # 获取标签单元格的点的个数
        total_x = 0
        total_y = 0  # 初始中心坐标的各项参数
        for j in range(0, m):
            total_x += dic_label["shapes"][i]["points"][j][0]
            total_y += dic_label["shapes"][i]["points"][j][1]
        average_x = int(total_x / m)
        average_y = int(total_y / m)  # 得到标签单元格的中心坐标
        label_cell.append([average_x, average_y, i])  # 将标签单元格的信息存储起来
    n = len(dic_polygon["shapes"])  # 获取单元格数目
    for p in range(0, n):  # 遍历每一个结构单元格
        polygon_content = []  # 储存单元格内容的数组，形式与label_cell一致，就是将满足要求的label_cell里的内容放进来
        b = len(dic_polygon["shapes"][p]["points"])  # 获取结构单元格的点的个数
        coords = []
        for q in range(0, b):
            coords.append(dic_polygon["shapes"][p]["points"][q])
        poly = Polygon(coords)  # 得到结构单元格的polygon

        for i in range(0, k):  # 遍历每一个标签是否在该单元格内
            p1 = Point(label_cell[i][0], label_cell[i][1])
            if poly.contains(p1):
                polygon_content.append(label_cell[i])  # 如果标签中心点在该单元格内，则将内容对应的序号放进数组

        c = len(polygon_content)
        if c == 1:
            dic_polygon["shapes"][p]["label"] += dic_label["shapes"][int(polygon_content[0][2])]["label"]
            # 如果只有一个标签在该单元格内，直接加入，不存在排序的问题
        elif c == 0:
            pass
        else:  # 多个标签被包含在单元格内
            def takeSecond(elem):
                return elem[1] + 0.05 * elem[0]  # 获取列表的第二个元素和第一个元素

            polygon_content.sort(key=takeSecond)  # 以中心坐标y的值为主进行排序，x坐标作为加权进行辅助排序
            for i in range(0, c):
                dic_polygon["shapes"][p]["label"] += dic_label["shapes"][int(polygon_content[i][2])]["label"]
                if i < c - 1:
                    if polygon_content[i + 1][1] - polygon_content[i][1] >= 10:
                        dic_polygon["shapes"][p]["label"] += "\n"
                    else:
                        dic_polygon["shapes"][p]["label"] += " "
    with open(polygon_file_path, 'w', encoding='utf-8') as f:  # 写入
        json.dump(dic_polygon, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':

    polygon_path = "X:/LAB/dataset/CurveTabSet/add/invoice/chinese_all-line/"  # 结构单元格文件夹地址
    label_path = "X:/LAB/dataset/CurveTabSet/add/content/"  # 标签文件夹地址
    shutil_path = "X:/LAB/dataset/CurveTabSet/add/error/"  # 若不存在对应的标签，将结构单元格移至该文件夹
    a = glob.glob(polygon_path + "*.json")  # 以结构单元格文件为基准进行遍历

    for name in tqdm(a, desc="add_label_to_polygon"):
        f_name, ext = os.path.splitext(name)
        base_name = os.path.basename(f_name)
        json_name = base_name + ".json"
        try:
            add_label(label_path + json_name, polygon_path + json_name)
        except Exception:
            shutil.move(polygon_path + base_name + ".json",
                        shutil_path)
            shutil.move(polygon_path + base_name + ".jpg",
                        shutil_path)

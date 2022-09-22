import json
import os
import glob
from tqdm import tqdm
from shapely.geometry import Point, Polygon

if __name__ == '__main__':
    TSR_path = 'X:/LAB/dataset/CurveTabSet/english_three-line/TSR_TCR_annotation/'
    TD_path = 'X:/LAB/dataset/CurveTabSet/english_three-line/TD_annotation/'
    a = glob.glob(TSR_path + "*.json")  # 以结构单元格文件为基准进行遍历

    for name in tqdm(a, desc="add_label_to_polygon"):
        f_name, ext = os.path.splitext(name)
        base_name = os.path.basename(f_name)
        json_name = base_name + ".json"
        with open(TSR_path + json_name, encoding='utf-8') as file:
            dic_TSR = json.load(file)
        with open(TD_path + json_name, encoding='utf-8') as file:
            dic_TD = json.load(file)

        k = len(dic_TSR["shapes"])  # 获取标签数目
        label_cell = []  # 存储标签信息的数组(中心横坐标，中心纵坐标，对应单元格内容序号)
        for i in range(0, k):  # 遍历每一个标签
            m = len(dic_TSR["shapes"][i]["points"])  # 获取标签单元格的点的个数
            total_x = 0
            total_y = 0  # 初始中心坐标的各项参数
            for j in range(0, m):
                total_x += dic_TSR["shapes"][i]["points"][j][0]
                total_y += dic_TSR["shapes"][i]["points"][j][1]
            average_x = int(total_x / m)
            average_y = int(total_y / m)  # 得到标签单元格的中心坐标
            label_cell.append([average_x, average_y, dic_TSR["shapes"][i]['group_id']])

        n = len(dic_TD["shapes"])  # 获取单元格数目
        for p in range(0, n):  # 遍历每一个结构单元格
            b = len(dic_TD["shapes"][p]["points"])  # 获取结构单元格的点的个数
            coords = []
            value = 0
            for q in range(0, b):
                coords.append(dic_TD["shapes"][p]["points"][q])
            poly = Polygon(coords)
            for i in range(0, k):  # 遍历每一个标签是否在该单元格内
                p1 = Point(label_cell[i][0], label_cell[i][1])
                if poly.contains(p1):
                    dic_TD["shapes"][p]["group_id"] = label_cell[i][2]
                    dic_TD["shapes"][p]["label"] = "table" + str(label_cell[i][2])
                    value = 1
                    break
            if value == 0:
                print(json_name)

        with open(TD_path + json_name, 'w', encoding='utf-8') as f:  # 写入
            json.dump(dic_TD, f, indent=4, ensure_ascii=False)

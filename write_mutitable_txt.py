import json
from tqdm import tqdm
import glob
import os

if __name__ == '__main__':

    path = "X:/LAB/CurveTabSet_2/en+ch_table/"
    a = glob.glob(path + "*.json")

    for name in tqdm(a, desc="muliti"):
        f_name, ext = os.path.splitext(name)
        base_name = os.path.basename(f_name)
        img_name = base_name + ".json"
        p = 0
        with open(path + img_name, encoding='utf-8') as file:
            dic = json.load(file)
        k = len(dic["shapes"])  # 获取单元格数目
        for i in range(0, k):
            if dic["shapes"][i]["group_id"] == 1:
                p += 1
        if p >= 1:
            with open("X:/LAB/CurveTabSet_2/multi-tables.txt", "a") as f:
                f.write("\n" + base_name + ".jpg")

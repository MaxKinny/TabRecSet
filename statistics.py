"""
    Make statistics on Chinese and English tables and draw corresponding histograms.
                include:table number,table row,table column,cell number,cell start row,cell end row
                        cell start column,cell end column,cell rowspan,cell colspan.
    Besides, can also calculate the maximum, minimum, mean, and standard deviation
    of each parameter
"""
import json
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

if __name__ == '__main__':
    path = "X:/LAB/CurveTabSet/chinese/image_json/"
    path_2 = "X:/LAB/CurveTabSet/english/image_json/"
    a = glob.glob(path + "*.json")
    a_2 = glob.glob(path_2 + "*.json")
    b = 0
    b_2 = 0
    table_nums = []  # 表格数量
    row_nums = []  # 行数
    col_nums = []  # 列数
    cell_nums = []  # 单元格数
    start_row = []
    end_row = []
    start_column = []
    end_column = []
    rows_nums = []  # 跨行数
    cols_nums = []  # 跨列数

    table_nums_2 = []  # 表格数量
    row_nums_2 = []  # 行数
    col_nums_2 = []  # 列数
    cell_nums_2 = []  # 单元格数
    start_row_2 = []
    end_row_2 = []
    start_column_2 = []
    end_column_2 = []
    rows_nums_2 = []  # 跨行数
    cols_nums_2 = []  # 跨列数
    print("中文图片数量:" + str(len(a)))
    print("英文图片数量:" + str(len(a_2)))

    for name in tqdm(a, desc="chinese_statistics"):
        f_name, ext = os.path.splitext(name)
        base_name = os.path.basename(f_name)
        img_name = base_name + ".json"
        try:
            table_num = 1
            with open(path + img_name, encoding='utf-8') as file:
                dic = json.load(file)
                k = len(dic["shapes"])  # 获取单元格数目
                for i in range(0, k):
                    start_row.append(int(dic["shapes"][i]["label"].split('-')[0]))
                    end_row.append(
                        int(dic["shapes"][i]["label"].split('-')[0]) + int(dic["shapes"][i]["label"].split('-')[2]) - 1)
                    start_column.append(int(dic["shapes"][i]["label"].split('-')[1]))
                    end_column.append(
                        int(dic["shapes"][i]["label"].split('-')[1]) + int(dic["shapes"][i]["label"].split('-')[3]) - 1)
                    rows_nums.append(int(dic["shapes"][i]["label"].split('-')[2]))
                    cols_nums.append(int(dic["shapes"][i]["label"].split('-')[3]))

            polygons = dic["shapes"]
            polygons_groupBy_groupID = {}
            for poly_ in polygons:
                key_ = poly_["group_id"]
                if polygons_groupBy_groupID.get(key_) is None:
                    polygons_groupBy_groupID[key_] = []
                polygons_groupBy_groupID[key_].append(poly_)

            table_nums.append(len(polygons_groupBy_groupID))  # 表格数

            for group_id, polygons in polygons_groupBy_groupID.items():
                cell_nums.append(len(polygons))
                row_num = 1
                col_num = 1
                for polygon_ in polygons:
                    if int(polygon_["label"].split('-')[0]) > row_num:
                        row_num = int(polygon_["label"].split('-')[0])
                    if int(polygon_["label"].split('-')[1]) > col_num:
                        col_num = int(polygon_["label"].split('-')[1])
                row_nums.append(row_num)
                col_nums.append(col_num)
                b += 1



        except Exception:
            print(base_name)

    for name in tqdm(a_2, desc="english_statistics"):
        f_name, ext = os.path.splitext(name)
        base_name = os.path.basename(f_name)
        img_name = base_name + ".json"
        try:
            table_num = 1
            with open(path_2 + img_name, encoding='utf-8') as file:
                dic = json.load(file)
                k = len(dic["shapes"])  # 获取单元格数目
                for i in range(0, k):
                    start_row_2.append(int(dic["shapes"][i]["label"].split('-')[0]))
                    end_row_2.append(
                        int(dic["shapes"][i]["label"].split('-')[0]) + int(dic["shapes"][i]["label"].split('-')[2]) - 1)
                    start_column_2.append(int(dic["shapes"][i]["label"].split('-')[1]))
                    end_column_2.append(
                        int(dic["shapes"][i]["label"].split('-')[1]) + int(dic["shapes"][i]["label"].split('-')[3]) - 1)
                    rows_nums_2.append(int(dic["shapes"][i]["label"].split('-')[2]))
                    cols_nums_2.append(int(dic["shapes"][i]["label"].split('-')[3]))

            polygons = dic["shapes"]
            polygons_groupBy_groupID = {}
            for poly_ in polygons:
                key_ = poly_["group_id"]
                if polygons_groupBy_groupID.get(key_) is None:
                    polygons_groupBy_groupID[key_] = []
                polygons_groupBy_groupID[key_].append(poly_)

            table_nums_2.append(len(polygons_groupBy_groupID))  # 表格数

            for group_id, polygons in polygons_groupBy_groupID.items():
                cell_nums_2.append(len(polygons))
                row_num = 1
                col_num = 1
                for polygon_ in polygons:
                    if int(polygon_["label"].split('-')[0]) > row_num:
                        row_num = int(polygon_["label"].split('-')[0])
                    if int(polygon_["label"].split('-')[1]) > col_num:
                        col_num = int(polygon_["label"].split('-')[1])
                row_nums_2.append(row_num)
                col_nums_2.append(col_num)
                b_2 += 1



        except Exception:
            print(base_name)

    print("中文表格数量:" + str(b))
    print("英文表格数量:" + str(b_2))

    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['figure.figsize'] = (12, 12)
    mpl.rcParams["font.size"] = 65
    ######################### 表格数 #######5################
    plt.hist([table_nums, table_nums_2], bins=7,
             rwidth=0.8,
             range=(1, 8),
             align='left', alpha=0.5, label=['chinese', 'english'], stacked=True, log=True)
    # for i in range(len(n1)):
    #     plt.text(bins[i], n1[i] * 1.01, int(n1[i]), fontsize=12, horizontalalignment="center")
    # for i in range(len(n2)):
    #     plt.text(bins[i], n2[i] * 1.01, int(n2[i]), fontsize=12, horizontalalignment="center")
    plt.xlabel('table number')
    plt.ylabel('number of picture')
    plt.legend(loc='upper right', frameon=False, borderpad=-0.4, labelspacing=-0.1, handlelength=1, handletextpad=0.1)
    plt.yticks()
    plt.xticks(range(8))
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    print("中文表格数：%f %f %f %f" % (max(table_nums), min(table_nums), np.mean(table_nums), np.std(table_nums, ddof=1)))
    print("英文表格数：%f %f %f %f" % (
        max(table_nums_2), min(table_nums_2), np.mean(table_nums_2), np.std(table_nums_2, ddof=1)))
    plt.tight_layout()  # 解决字体遮挡
    plt.savefig("table number.png", bbox_inches="tight")
    plt.show()

    ######################### 行数 #######################
    plt.hist([row_nums, row_nums_2], bins=7, rwidth=0.8,
             range=(1, 52),
             align='left', alpha=0.5, label=['chinese', 'english'], stacked=True, log=True)
    plt.xlabel('row number')
    plt.ylabel('number of table')
    plt.legend(loc='upper right', frameon=False, borderpad=-0.4, labelspacing=-0.1, handlelength=1, handletextpad=0.1)
    plt.yticks()
    plt.xticks([1,10,20,30,40,50])
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    print("中文行数：%f %f %f %f" % (max(row_nums), min(row_nums), np.mean(row_nums), np.std(row_nums, ddof=1)))
    print("英文行数：%f %f %f %f" % (
        max(row_nums_2), min(row_nums_2), np.mean(row_nums_2), np.std(row_nums_2, ddof=1)))
    plt.tight_layout()  # 解决字体遮挡
    plt.savefig("table row.png", bbox_inches="tight")
    plt.show()

    ######################### 列数 #######################
    plt.hist([col_nums, col_nums_2], bins=7, rwidth=0.8,
             range=(1, 28),
             align='left', alpha=0.5, label=['chinese', 'english'], stacked=True, log=True)
    plt.xlabel('column number')
    plt.ylabel('number of table')
    plt.legend(loc='upper right', frameon=False, borderpad=-0.4, labelspacing=-0.1, handlelength=1, handletextpad=0.1)
    plt.yticks()
    plt.xticks([1,5,10,15,20,25])
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    print("中文列数：%f %f %f %f" % (max(col_nums), min(col_nums), np.mean(col_nums), np.std(col_nums, ddof=1)))
    print("英文列数：%f %f %f %f" % (
        max(col_nums_2), min(col_nums_2), np.mean(col_nums_2), np.std(col_nums_2, ddof=1)))
    plt.tight_layout()  # 解决字体遮挡
    plt.savefig("table column.png", bbox_inches="tight")
    plt.show()

    ######################### 单元格数 #######################
    plt.hist([cell_nums, cell_nums_2], bins=7, rwidth=0.8,
             range=(1, 351),
             align='left', alpha=0.5, label=['chinese', 'english'], stacked=True, log=True)
    plt.xlabel('cell number')
    plt.ylabel('number of table')
    plt.legend(loc='upper right', frameon=False, borderpad=-0.4, labelspacing=-0.1, handlelength=1, handletextpad=0.1)
    plt.yticks()
    plt.xticks([1,100,200,300])
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    print("中文单元格数：%f %f %f %f" % (max(cell_nums), min(cell_nums), np.mean(cell_nums), np.std(cell_nums, ddof=1)))
    print("英文单元格数：%f %f %f %f" % (
        max(cell_nums_2), min(cell_nums_2), np.mean(cell_nums_2), np.std(cell_nums_2, ddof=1)))
    plt.tight_layout()  # 解决字体遮挡
    plt.savefig("cell number.png", bbox_inches="tight")
    plt.show()

    ######################### 起始行 #######################
    plt.hist([start_row, start_row_2], bins=7, rwidth=0.8,
             range=(1, 52),
             align='left', alpha=0.5, label=['chinese', 'english'], stacked=True, log=True)
    plt.xlabel('start row number')
    plt.ylabel('number of cell')
    plt.legend(loc='upper right', frameon=False, borderpad=-0.4, labelspacing=-0.1, handlelength=1, handletextpad=0.1)
    plt.yticks()
    plt.xticks([1,10,20,30,40,50])
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    print("中文起始行：%f %f %f %f" % (max(start_row), min(start_row), np.mean(start_row), np.std(start_row, ddof=1)))
    print("英文起始行：%f %f %f %f" % (
        max(start_row_2), min(start_row_2), np.mean(start_row_2), np.std(start_row_2, ddof=1)))
    plt.tight_layout()  # 解决字体遮挡
    plt.savefig("cell start row.png", bbox_inches="tight")
    plt.show()
    ######################### 结束行 #######################
    plt.hist([end_row, end_row_2], bins=7, rwidth=0.8,
             range=(1, 52),
             align='left', alpha=0.5, label=['chinese', 'english'], stacked=True, log=True)
    plt.xlabel('end row number')
    plt.ylabel('number of cell')
    plt.legend(loc='upper right', frameon=False, borderpad=-0.4, labelspacing=-0.1, handlelength=1, handletextpad=0.1)
    plt.yticks()
    plt.xticks([1,10,20,30,40,50])
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    print("中文结束行：%f %f %f %f" % (max(end_row), min(end_row), np.mean(end_row), np.std(end_row, ddof=1)))
    print("英文结束行：%f %f %f %f" % (
        max(end_row_2), min(end_row_2), np.mean(end_row_2), np.std(end_row_2, ddof=1)))
    plt.tight_layout()  # 解决字体遮挡
    plt.savefig("cell end row.png", bbox_inches="tight")
    plt.show()

    ######################### 起始列 #######################
    plt.hist([start_column, start_column_2], bins=7, rwidth=0.8,
             range=(1, 28),
             align='left', alpha=0.5, label=['chinese', 'english'], stacked=True, log=True)
    plt.xlabel('start column number')
    plt.ylabel('number of cell')
    plt.legend(loc='upper right', frameon=False, borderpad=-0.4, labelspacing=-0.1, handlelength=1, handletextpad=0.1)
    plt.yticks()
    plt.xticks([1,5,10,15,20,25])
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    print("中文起始列：%f %f %f %f" % (
        max(start_column), min(start_column), np.mean(start_column), np.std(start_column, ddof=1)))
    print("英文起始列：%f %f %f %f" % (
        max(start_column_2), min(start_column_2), np.mean(start_column_2), np.std(start_column_2, ddof=1)))
    plt.tight_layout()  # 解决字体遮挡
    plt.savefig("cell start column.png", bbox_inches="tight")
    plt.show()

    ######################### 结束列 #######################
    plt.hist([end_column, end_column_2], bins=7, rwidth=0.8,
             range=(1, 28),
             align='left', alpha=0.5, label=['chinese', 'english'], stacked=True, log=True)
    plt.xlabel('end column number')
    plt.ylabel('number of cell')
    plt.legend(loc='upper right', frameon=False, borderpad=-0.4, labelspacing=-0.1, handlelength=1, handletextpad=0.1)
    plt.yticks()
    plt.xticks([1,5,10,15,20,25])
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    print("中文结束列：%f %f %f %f" % (max(end_column), min(end_column), np.mean(end_column), np.std(end_column, ddof=1)))
    print("英文结束列：%f %f %f %f" % (
        max(end_column_2), min(end_column_2), np.mean(end_column_2), np.std(end_column_2, ddof=1)))
    plt.tight_layout()  # 解决字体遮挡
    plt.savefig("cell end column.png", bbox_inches="tight")
    plt.show()

    ######################### 跨行 #######################
    plt.hist([rows_nums, rows_nums_2], bins=7, rwidth=0.8,
             range=(1, 42),
             align='left', alpha=0.5, label=['chinese', 'english'], stacked=True, log=True)
    plt.xlabel('rowspan number')
    plt.ylabel('number of cell')
    plt.legend(loc='upper right', frameon=False, borderpad=-0.4, labelspacing=-0.1, handlelength=1, handletextpad=0.1)
    plt.yticks()
    plt.xticks([1,10,20,30,40])
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    print("中文跨行：%f %f %f %f" % (max(rows_nums), min(rows_nums), np.mean(rows_nums), np.std(rows_nums, ddof=1)))
    print("英文跨行：%f %f %f %f" % (
        max(rows_nums_2), min(rows_nums_2), np.mean(rows_nums_2), np.std(rows_nums_2, ddof=1)))
    plt.tight_layout()  # 解决字体遮挡
    plt.savefig("cell rowspan.png", bbox_inches="tight")
    plt.show()

    ######################### 跨列 #######################
    plt.hist([cols_nums, cols_nums_2], bins=7, rwidth=0.8,
             range=(1, 25),
             align='left', alpha=0.5, label=['chinese', 'english'], stacked=True, log=True)
    plt.xlabel('colspan number')
    plt.ylabel('number of cell')
    plt.legend(loc='upper right', frameon=False, borderpad=-0.4, labelspacing=-0.1, handlelength=1, handletextpad=0.1)
    plt.yticks()
    plt.xticks([1,5,10,15,20])
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    print("中文跨列：%f %f %f %f" % (max(cols_nums), min(cols_nums), np.mean(cols_nums), np.std(cols_nums, ddof=1)))
    print("英文跨列：%f %f %f %f" % (
        max(cols_nums_2), min(cols_nums_2), np.mean(cols_nums_2), np.std(cols_nums_2, ddof=1)))
    plt.tight_layout()  # 解决字体遮挡
    plt.savefig("cell colspan.png", bbox_inches="tight")
    plt.show()

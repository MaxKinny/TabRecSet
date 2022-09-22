import json
import os
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from collections import Counter
import nltk


def is_contains_english(strs):
    for word in strs:
        if (u'\u0041' <= word <= u'\u005a') or (u'\u0061' <= word <= u'\u007a'):
            return False
    return True


def is_contains_numbers(strs):
    for word in strs:
        if word in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
            return False
    return True


if __name__ == '__main__':
    path = "X:/LAB/dataset/CurveTabSet/chinese_all-line/TD_annotation/"
    path_2 = "X:/LAB/dataset/CurveTabSet/english_all-line/TD_annotation/"
    a = glob.glob(path + "*.json")
    a_2 = glob.glob(path_2 + "*.json")

    chinese_character_dict = Counter()
    chinese_content_long = []
    chinese_word_dict = Counter()
    english_character_dict = Counter()
    english_content_long = []
    english_word_dict = Counter()
    chinese_point = []
    english_point = []
    all_character_dict = Counter()
    content = ''

    for name in tqdm(a, desc="chinese_statistics"):
        f_name, ext = os.path.splitext(name)
        base_name = os.path.basename(f_name)
        img_name = base_name + ".json"
        try:
            with open(path + img_name, encoding='utf-8') as file:
                dic = json.load(file)
                k = len(dic["shapes"])  # 获取单元格数目
                for i in range(0, k):
                    # chinese_cell_content = str(dic["shapes"][i]["label"].split('-', 4)[4])
                    # chinese_content_long.append(len(chinese_cell_content))
                    # chinese_character_dict.update(chinese_cell_content)
                    # all_character_dict.update(chinese_cell_content)
                    chinese_point.append(len(dic["shapes"][i]["points"]))
                    # tokens = nltk.word_tokenize(chinese_cell_content)
                    # chinese_word_dict.update(tokens)

        except Exception:
            print(base_name)

    for name in tqdm(a_2, desc="english_statistics"):
        f_name, ext = os.path.splitext(name)
        base_name = os.path.basename(f_name)
        img_name = base_name + ".json"
        try:
            with open(path_2 + img_name, encoding='utf-8') as file:
                dic = json.load(file)
                k = len(dic["shapes"])  # 获取单元格数目
                for i in range(0, k):
                    # english_cell_content = str(dic["shapes"][i]["label"].split('-', 4)[4])
                    # chinese_content_long.append(len(english_cell_content))
                    # english_character_dict.update(english_cell_content)
                    # all_character_dict.update(english_cell_content)
                    english_point.append(len(dic["shapes"][i]["points"]))

                    # english_cell_word = english_cell_content.split(' ')
                    # english_word_dict.update(english_cell_word)

        except Exception:
            print(base_name)
    a = 24
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['figure.figsize'] = (a, a * 0.618)
    mpl.rcParams["font.size"] = 80
    content = ''

    ######################### 文本长度 #######5################
    # plt.hist(chinese_content_long, bins=69,
    #          rwidth=0.8,
    #          range=(1, 70),
    #          align='left', alpha=0.5, stacked=True, log=True)
    # plt.xlabel('length of cell content')
    # plt.ylabel('cell number')
    # # plt.legend(fontsize=80)
    # plt.yticks([10, 100, 1000, 10000,100000])
    # plt.xticks([1, 10, 20, 30, 40, 50, 60, 70])
    # ax = plt.gca()  # 获得坐标轴的句柄
    # ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    # ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    # ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    # plt.tight_layout()  # 解决字体遮挡
    # plt.savefig("content length.png", bbox_inches="tight")
    # plt.show()

    ######################### 顶点个数 #######5################
    plt.hist([chinese_point,english_point], bins=9,
             rwidth=0.8,
             range=(4, 13),
             align='left', alpha=0.5, label=['chinese', 'english'],stacked=True, log=True)
    plt.xlabel('number of polygon vertex')
    plt.ylabel('number of table')
    plt.legend(loc='upper right', frameon=False, borderpad=-0.4, labelspacing=-0.1, handlelength=1, handletextpad=0.1)
    plt.yticks([10,100, 1000, 10000])
    plt.xticks([4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    plt.tight_layout()  # 解决字体遮挡
    plt.savefig("tabel point number.png", bbox_inches="tight")
    plt.show()

    ######################### 英文字符 #######5################
    # plt.rcParams['figure.figsize'] = (a, a * 0.4)
    # english_character_dict_keys = []
    # english_character_dict_values = []
    #
    # for a in sorted(english_character_dict, key=english_character_dict.__getitem__, reverse=True):
    #     if len(english_character_dict_keys) <= 60:
    #         if a != " " and a != "" and a != "\n":
    #             english_character_dict_keys.append(a)
    #             english_character_dict_values.append(int(english_character_dict[a]))
    # y = english_character_dict_values
    # x = range(len(english_character_dict_keys))
    #
    # plt.bar(x, english_character_dict_values, tick_label=english_character_dict_keys, alpha=0.5)
    # plt.yticks([40000, 80000, 120000, 160000], size=30)
    # plt.xticks(size=30)
    # ax = plt.gca()  # 获得坐标轴的句柄
    # ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    # ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    # ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    # plt.tight_layout()  # 解决字体遮挡
    # plt.savefig("english character.png", bbox_inches="tight")
    # plt.show()

    ######################### 英文单词 #######5################
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.2)
    # english_word_dict_keys = []
    # english_word_dict_values = []
    #
    # for a in sorted(english_word_dict, key=english_word_dict.__getitem__, reverse=True):
    #     if len(english_word_dict_keys) <= 35:
    #         if a != "" and len(a) > 1 and a != "###":
    #             english_word_dict_keys.append(a)
    #             english_word_dict_values.append(int(english_word_dict[a]))
    # y = english_word_dict_values
    # x = range(len(english_word_dict_keys))
    #
    # plt.bar(x, english_word_dict_values, tick_label=english_word_dict_keys, alpha=0.5, log=True)
    # plt.yticks([10,100,1000,10000],size=30)
    # plt.xticks(size=30, rotation=60)
    # plt.xlabel("words", size=30)
    # plt.ylabel("word number", size=30)
    # ax = plt.gca()  # 获得坐标轴的句柄
    # ax.spines['bottom'].set_linewidth(2)  ###设置底部坐标轴的粗细
    # ax.spines['left'].set_linewidth(2)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(2)  ###设置右边坐标轴的粗细
    # ax.spines['top'].set_linewidth(2)  ####设置上部坐标轴的粗细
    # plt.tight_layout()  # 解决字体遮挡
    # plt.savefig("english word.png", bbox_inches="tight")
    # plt.show()

    ######################### 中文字符 #######5################
    # plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
    #
    # chinese_character_dict_keys = []
    # chinese_character_dict_values = []
    #
    # for a in sorted(chinese_character_dict, key=chinese_character_dict.__getitem__, reverse=True):
    #     if len(chinese_character_dict_keys) <= 50 and a != " " and a != "" and a != "\n":
    #         chinese_character_dict_keys.append(a)
    #         chinese_character_dict_values.append(int(chinese_character_dict[a]))
    # y = chinese_character_dict_values
    # x = range(len(chinese_character_dict_keys))
    #
    # plt.bar(x, chinese_character_dict_values, tick_label=chinese_character_dict_keys, alpha=0.5)
    # plt.yticks([5000, 10000, 15000, 20000, 25000, 30000], size=30)
    # plt.xticks(size=30)
    # ax = plt.gca()  # 获得坐标轴的句柄
    # ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    # ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    # ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    # plt.tight_layout()  # 解决字体遮挡
    # plt.savefig("chinese character.png", bbox_inches="tight")
    # plt.show()

    ######################### 中文单词 #######5################
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.2)
    # plt.rcParams['font.family'] = ['SimSun']  # 用来正常显示中文标签
    # chinese_word_dict_keys = []
    # chinese_word_dict_values = []
    #
    # for a in sorted(chinese_word_dict, key=chinese_word_dict.__getitem__, reverse=True):
    #     if len(chinese_word_dict_keys) <= 35:
    #         if is_contains_english(a) and is_contains_numbers(a):
    #             if a != "" and len(a) > 1 and a != "###" and len(a) < 4 and a != "''" and a != '……' and a != '--':
    #                 chinese_word_dict_keys.append(a)
    #                 chinese_word_dict_values.append(int(chinese_word_dict[a]))
    # y = chinese_word_dict_values
    # x = range(len(chinese_word_dict_keys))
    #
    # plt.bar(x, chinese_word_dict_values, tick_label=chinese_word_dict_keys, alpha=0.5)
    # plt.yticks([50, 100, 150, 200], size=30)
    # plt.xticks(size=25, rotation=60)
    # plt.xlabel("words", size=30, family='Times New Roman')
    # plt.ylabel("word number", size=30, family='Times New Roman')
    # ax = plt.gca()  # 获得坐标轴的句柄
    # ax.spines['bottom'].set_linewidth(2)  ###设置底部坐标轴的粗细
    # ax.spines['left'].set_linewidth(2)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(2)  ###设置右边坐标轴的粗细
    # ax.spines['top'].set_linewidth(2)  ####设置上部坐标轴的粗细
    # plt.tight_layout()  # 解决字体遮挡
    # plt.savefig("中文单词.png", bbox_inches="tight")
    # plt.show()

    # content += "#################中文_______单词###############\n\n"
    # for i in range(len(chinese_word_dict_values)):
    #     content += "字符：" + str(chinese_word_dict_keys[i]) + "   个数:   " + str(chinese_word_dict_values[i]) + "  "
    #     if (i + 1) % 5 == 0:
    #         content += "\n"
    # content += "\n"

    ###########################英文----------汉字########################
    # plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.4)
    # english_4_dict_keys = []
    # english_4_dict_values = []
    # for a in sorted(english_character_dict, key=english_character_dict.__getitem__, reverse=True):
    #         if '\u4e00' <= a <= '\u9fa5':
    #             english_4_dict_keys.append(a)
    #             english_4_dict_values.append(int(english_character_dict[a]))
    #
    # y = english_4_dict_values
    # x = range(len(english_4_dict_keys))
    #
    # plt.bar(x, english_4_dict_values, tick_label=english_4_dict_keys, alpha=0.5)
    # plt.yticks(size=30)
    # plt.xticks(size=25)
    # ax = plt.gca()  # 获得坐标轴的句柄
    # ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    # ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    # ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    # plt.tight_layout()  # 解决字体遮挡
    # plt.savefig("英文汉字.png", bbox_inches="tight")
    # plt.show()
    #
    # content += "#################英文_______汉字###############\n\n"
    # for i in range(len(english_4_dict_values)):
    #     content += "字符：" + str(english_4_dict_keys[i]) + "   个数:   " + str(english_4_dict_values[i]) + "  "
    #     if (i + 1) % 5 == 0:
    #         content += "\n"
    # content += "\n"

    ###########################英文----------小写字符########################
    # english_1_dict_keys = []
    # english_1_dict_values = []
    # for a in sorted(english_character_dict.items(), key=lambda e: e[0]):
    #     if u'\u0061' <= a[0] <= u'\u007a':
    #         english_1_dict_keys.append(a[0])
    #         english_1_dict_values.append(int(a[1]))
    ###########################英文----------大写字符########################
    # english_2_dict_keys = []
    # english_2_dict_values = []
    #
    # for a in sorted(english_character_dict.items(), key=lambda e: e[0]):
    #     if u'\u0041' <= a[0] <= u'\u005a':
    #         english_2_dict_keys.append(a[0])
    #         english_2_dict_values.append(int(a[1]))
    #########################英文----------数字############################
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.618)
    # english_3_dict_keys = []
    # english_3_dict_values = []
    #
    # for a in sorted(english_character_dict.items(), key=lambda e: e[0]):
    #     if a[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
    #         english_3_dict_keys.append(a[0])
    #         english_3_dict_values.append(int(a[1]))

    #########################英文----------特殊字符############################
    # english_5_dict_keys = []
    # english_5_dict_values = []
    # for a in sorted(english_character_dict.items(), key=lambda e: e[0]):
    #     if u'\u0061' <= a[0] <= u'\u007a':
    #         pass
    #     elif a[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
    #         pass
    #     elif u'\u0041' <= a[0] <= u'\u005a':
    #         pass
    #     elif a[0] == " ":
    #         pass
    #     elif a[0] == "":
    #         pass
    #     elif a[0] == "\n":
    #         pass
    #     elif a[0] == "\t":
    #         pass
    #     else:
    #         english_5_dict_keys.append(a[0])
    #         english_5_dict_values.append(int(a[1]))
    # content += "#################英文_______特殊字符###############\n\n"
    # for i in range(len(english_5_dict_values)):
    #     content += "字符：" + str(english_5_dict_keys[i]) + "   个数:   " + str(english_5_dict_values[i]) + "  "
    #     if (i + 1) % 5 == 0:
    #         content += "\n"
    # content += "\n"

    ###########################中文----------小写字符########################
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.4)
    # chinese_1_dict_keys = []
    # chinese_1_dict_values = []
    # for a in sorted(chinese_character_dict.items(), key=lambda e: e[0]):
    #     if u'\u0061' <= a[0] <= u'\u007a':
    #         chinese_1_dict_keys.append(a[0])
    #         chinese_1_dict_values.append(int(a[1]))
    # width = 0.4
    # x1 = np.arange(len(chinese_1_dict_keys))
    # x2 = np.arange(len(english_1_dict_keys))
    # plt.bar(x1 - width, chinese_1_dict_values, width, label="chinese",alpha=0.5, log=True)
    # plt.bar(x2, english_1_dict_values, width, alpha=0.5, label="english",
    #         log=True)
    #
    # plt.yticks([100, 1000, 10000, 100000], size=30)
    # plt.xticks([i - width / 2 for i in x1], chinese_1_dict_keys, size=30)
    # plt.legend(fontsize=30)
    # ax = plt.gca()  # 获得坐标轴的句柄
    # ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    # ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    # ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    # plt.tight_layout()  # 解决字体遮挡
    # plt.savefig("小写字母.png", bbox_inches="tight")
    # plt.show()

    ###########################中文----------大写字符########################
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.4)
    # chinese_2_dict_keys = []
    # chinese_2_dict_values = []
    # for a in sorted(chinese_character_dict.items(), key=lambda e: e[0]):
    #     if u'\u0041' <= a[0] <= u'\u005a':
    #         chinese_2_dict_keys.append(a[0])
    #         chinese_2_dict_values.append(int(a[1]))
    # width = 0.4
    # x1 = np.arange(len(chinese_2_dict_keys))
    # x2 = np.arange(len(english_2_dict_keys))
    # plt.bar(x1 - width, chinese_2_dict_values, width, label="chinese", alpha=0.5, log=True)
    # plt.bar(x2, english_2_dict_values, width, alpha=0.5, label="english",
    #         log=True)
    # plt.yticks([100, 1000, 10000, 100000], size=30)
    # plt.xticks([i - width / 2 for i in x1], chinese_2_dict_keys, size=30)
    # plt.legend(fontsize=30)
    # ax = plt.gca()  # 获得坐标轴的句柄
    # ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    # ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    # ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    # plt.tight_layout()  # 解决字体遮挡
    # plt.savefig("大写字母.png", bbox_inches="tight")
    # plt.show()

    ###########################中文----------数字########################
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.618)
    # chinese_3_dict_keys = []
    # chinese_3_dict_values = []
    # for a in sorted(chinese_character_dict.items(), key=lambda e: e[0]):
    #     if a[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
    #         chinese_3_dict_keys.append(a[0])
    #         chinese_3_dict_values.append(int(a[1]))
    # width = 0.4
    # x1 = np.arange(len(chinese_3_dict_keys))
    # x2 = np.arange(len(english_3_dict_keys))
    # plt.bar(x1 - width, chinese_3_dict_values, width, label="chinese", alpha=0.5, log=True)
    # plt.bar(x2, english_3_dict_values, width, alpha=0.5, label="english",
    #         log=True)
    # plt.yticks([1000, 10000, 100000], size=30)
    # plt.xticks([i - width / 2 for i in x1], chinese_3_dict_keys, size=30)
    # plt.legend(fontsize=30)
    # ax = plt.gca()  # 获得坐标轴的句柄
    # ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    # ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    # ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    # plt.tight_layout()  # 解决字体遮挡
    # plt.savefig("数字.png", bbox_inches="tight")
    # plt.show()

    ###########################中文----------汉字____统计常用字符########################
    # words = []
    # with open("500.txt", encoding="utf-8") as f:
    #     for line in f:
    #         words.append(line.replace('\n', ''))
    #
    # plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.4)
    # chinese_4_dict_values = []
    # for i in range(len(words)):
    #     chinese_4_dict_values.append(0)
    # k = 0
    # for a in words:
    #     if a in chinese_character_dict:
    #         chinese_4_dict_values[k] = int(chinese_character_dict[a])
    #     k+=1
    #
    # for i in range(len(words)):
    #     content += str(words[i]) + "   " + str(chinese_4_dict_values[i]) + "\n"

    ##########################中文----------汉字########################
    # plt.rcParams['font.family'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.4)
    # chinese_4_dict_keys = []
    # chinese_4_dict_values = []
    # for a in sorted(chinese_character_dict, key=chinese_character_dict.__getitem__, reverse=True):
    #     if len(chinese_4_dict_keys) <= 40:
    #         if '\u4e00' <= a <= '\u9fa5':
    #             chinese_4_dict_keys.append(a)
    #             chinese_4_dict_values.append(int(chinese_character_dict[a]))
    #
    # y = chinese_4_dict_values
    # x = range(len(chinese_4_dict_keys))
    #
    # plt.bar(x, chinese_4_dict_values, tick_label=chinese_4_dict_keys, alpha=0.5)
    # plt.yticks(size=30)
    # plt.xticks(size=25)
    # plt.xlabel("words")
    # plt.ylabel("word number")
    # ax = plt.gca()  # 获得坐标轴的句柄
    # ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    # ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    # ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    # plt.tight_layout()  # 解决字体遮挡
    # plt.savefig("中文汉字.png", bbox_inches="tight")
    # plt.show()

    ###########################中文----------特殊字符########################
    # plt.rcParams['font.family'] = ['SimHei']
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.4)
    # chinese_5_dict_keys = []
    # chinese_5_dict_values = []
    # for a in sorted(chinese_character_dict.items(), key=lambda e: e[0]):
    #     if u'\u0061' <= a[0] <= u'\u007a':
    #         pass
    #     elif a[0] in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
    #         pass
    #     elif u'\u0041' <= a[0] <= u'\u005a':
    #         pass
    #     elif a[0] == " ":
    #         pass
    #     elif a[0] == "":
    #         pass
    #     elif a[0] == "\n":
    #         pass
    #     elif a[0] == "\t":
    #         pass
    #     elif '\u4e00' <= a[0] <= '\u9fa5':
    #         pass
    #     else:
    #         chinese_5_dict_keys.append(a[0])
    #         chinese_5_dict_values.append(int(a[1]))
    # content += "#################中文_______特殊字符###############\n\n"
    # for i in range(len(chinese_5_dict_values)):
    #     content += "字符：" + str(chinese_5_dict_keys[i]) + "   个数:   " + str(chinese_5_dict_values[i]) + "  "
    #     if (i + 1) % 5 == 0:
    #         content += "\n"
    # content += "\n"
    # width = 0.4
    # x1 = np.arange(len(chinese_5_dict_keys))
    # x2 = np.arange(len(english_4_dict_keys))
    # plt.bar(x1 - width, chinese_5_dict_values, width, label="chinese", alpha=0.5, log=True)
    # plt.bar(x2, english_4_dict_values, width, alpha=0.5, label="english",
    #         log=True)
    # plt.yticks([1, 10, 100, 1000, 10000, 100000], size=30)
    # plt.xticks([i - width / 2 for i in x1], chinese_5_dict_keys, size=30)
    # plt.legend(fontsize=30)
    # ax = plt.gca()  # 获得坐标轴的句柄
    # ax.spines['bottom'].set_linewidth(3)  ###设置底部坐标轴的粗细
    # ax.spines['left'].set_linewidth(3)  ####设置左边坐标轴的粗细
    # ax.spines['right'].set_linewidth(3)  ###设置右边坐标轴的粗细
    # ax.spines['top'].set_linewidth(3)  ####设置上部坐标轴的粗细
    # plt.tight_layout()  # 解决字体遮挡
    # plt.savefig("特殊字符.png", bbox_inches="tight")
    # plt.show()

    # ######################全部-----------小写#########################
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.4)
    # all_1_dict_keys = []
    # all_1_dict_values = []
    # for a in sorted(all_character_dict, key=all_character_dict.__getitem__, reverse=True):
    #     if u'\u0061' <= a <= u'\u007a':
    #         all_1_dict_keys.append(a)
    #         all_1_dict_values.append(int(all_character_dict[a]))
    # ######################全部-----------大写#########################
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.4)
    # all_2_dict_keys = []
    # all_2_dict_values = []
    # for a in sorted(all_character_dict, key=all_character_dict.__getitem__, reverse=True):
    #     if u'\u0041' <= a <= u'\u005a':
    #         all_2_dict_keys.append(a)
    #         all_2_dict_values.append(int(all_character_dict[a]))
    # ######################全部-----------数字#########################
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.618)
    # all_3_dict_keys = []
    # all_3_dict_values = []
    # for a in sorted(all_character_dict, key=all_character_dict.__getitem__, reverse=True):
    #     if a in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
    #         all_3_dict_keys.append(a)
    #         all_3_dict_values.append(int(all_character_dict[a]))
    #
    # ######################全部-----------汉字#########################
    # plt.rcParams['font.family'] = ['SimHei']
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.4)
    # all_4_dict_keys = []
    # all_4_dict_values = []
    # for a in sorted(all_character_dict, key=all_character_dict.__getitem__, reverse=True):
    #     if len(all_4_dict_keys) <= 40:
    #         if '\u4e00' <= a <= '\u9fa5':
    #             all_4_dict_keys.append(a)
    #             all_4_dict_values.append(int(all_character_dict[a]))
    #
    # ###########################全部----------特殊字符########################
    # plt.rcParams['font.family'] = ['SimHei']
    # plt.rcParams['figure.figsize'] = (24, 24 * 0.4)
    # all_5_dict_keys = []
    # all_5_dict_values = []
    # for a in sorted(all_character_dict, key=all_character_dict.__getitem__, reverse=True):
    #     if len(all_5_dict_keys) <= 25:
    #         if u'\u0061' <= a <= u'\u007a':
    #             pass
    #         elif a in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]:
    #             pass
    #         elif u'\u0041' <= a <= u'\u005a':
    #             pass
    #         elif a == " ":
    #             pass
    #         elif a == "":
    #             pass
    #         elif a == "\n":
    #             pass
    #         elif '\u4e00' <= a <= '\u9fa5':
    #             pass
    #         else:
    #             all_5_dict_keys.append(a)
    #             all_5_dict_values.append(int(all_character_dict[a]))
    # with open("yingwenteshu.txt",'w', encoding='utf-8') as f:
    #     f.write(content)

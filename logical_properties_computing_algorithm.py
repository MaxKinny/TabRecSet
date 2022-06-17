"""
    Generate table structure attributes in labelme format based on refined labelme 
    cell segmentation annotations and save to json files.
"""

import os, sys

project_path = os.path.dirname(os.path.realpath(__file__))
for _ in range(1):  # change the number, e.g. subfolder -> 1, subsubfolder -> 2
    project_path = os.path.dirname(project_path)
sys.path.insert(0, project_path)  # add project folder
os.chdir(project_path)  # change current folder to project folder

import traceback
import shutil
import json
from typing import List, Tuple
import glob, cv2
from tqdm import tqdm
import numpy as np
import math
import torch.nn.functional as F
import torch
import itertools
import torch.nn as nn
from torch.autograd import Variable

def add_filename_suffix(file_with_path: str, suffix: str) -> str:
    """/path/to/file/<filename>.<format> --> /path/to/file/<filename>_<suffix>.<format>
        or
        <filename>.<format> --> <filename>_<suffix>.<format>

    Args:
        file_with_path (str): [description]
        suffix (str): [description]

    Returns:
        str: [description]
    """
    file_path, filename_with_ext = os.path.split(file_with_path)
    filename, file_ext = os.path.splitext(filename_with_ext)
    res = os.path.join(file_path, filename + suffix + file_ext)
    return res


def change_file_format(file_with_path: str, new_format: str) -> str:
    """/path/to/file/<filename>.<old_format> --> /path/to/file/<filename>.<new_format>
        or
        <filename>.<old_format> --> <filename>.<new_format>
        e.g.
        file_with_path="/a/b/c/x.jpg", new_format=".png"
    """
    file_path, filename_with_ext = os.path.split(file_with_path)
    filename, _ = os.path.splitext(filename_with_ext)
    res = os.path.join(file_path, filename + new_format)
    return res

class Cell():
    """A class of table cell stores all possible raw info. on original image.
    """

    def __init__(self,
                 id,
                 upleft_point,
                 downright_point,
                 img_shape,
                 mask: np.ndarray = None,
                 content=None,
                 polygon=None,
                 rowspan=None,
                 colspan=None,
                 row_index=None,
                 colum_index=None):
        """
        Args:
            id (Integer): differentiate and identify each cell
            upleft_point (List): cell bbox's [x_min, y_min]
            downright_point (List): cell bbox's [x_max, y_max]
            img_shape (Tuple): shape of original image (Height, Width)
            mask (np.ndarray of bool): cell region mask on original image
            content (String): cell content
            polygon (List[List]): sorted [[w1, h1], [w2, h2] ...]
        """
        self.id = id

        self.colspan = colspan
        self.rowspan = rowspan
        self.row_index = row_index
        self.colum_index = colum_index

        self.downright_row_index = None

        self.img_shape = img_shape

        self.content = content

        if mask is None:
            if polygon is not None:
                self.mask: np.ndarray = self._mask_from_polygon(polygon)
                # TODO can add WarpRectContour member
            else:
                raise Exception("No enough info. for constructing Cell object!")
        else:
            self.mask: np.ndarray = mask

        if polygon is None:
            if mask is not None:
                self.polygon = self._polygon_from_mask(mask)
                # TODO can add WarpRectContour member
            else:
                raise Exception("No enough info. for constructing Cell object!")
        else:
            self.polygon = polygon

        if (upleft_point is not None) and (downright_point is not None):
            self.upleft = upleft_point
            self.downright = downright_point
        else:
            self.upleft, self.downright = self._points_from_polygon(self.polygon)

        self.mask_contour = None

    def _points_from_polygon(self, polygon: List) -> Tuple[int, int]:
        up_left, _, down_right, _ = tuple(contour_sorted_bbox(polygon))
        return up_left, down_right

    def get_mask_img(self):
        return self.mask.astype('uint8') * 255

    def has_main_property(self) -> bool:
        return self.colum_index is not None \
             and self.row_index is not None \
                 and self.rowspan is not None \
                     and self.colspan is not None

    def _mask_from_polygon(self, polygon: List) -> np.ndarray:
        """Generate mask from polygon.
        Args:
            polygon (List[List]): sorted [[w1, h1], [w2, h2] ...]
        Returns:
            (np.ndarray of bool): cell region mask on original image
        """
        img = np.zeros(self.img_shape, np.uint8)
        polygon_ndarray = np.array(polygon, dtype="int")
        mask_ = cv2.fillConvexPoly(img, polygon_ndarray, 255)
        return mask_ == 255

    def _polygon_from_mask(self, mask) -> List[List]:
        """Generate polygon from mask.
        Args:
            mask (np.ndarray of bool): cell region mask on original image
        Raises:
            Exception: No contour found then polygon will not be generated.
        Returns:
            List[List]: sorted [[w1, h1], [w2, h2] ...]
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = mask.astype('uint8') * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.dilate(mask, kernel)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 1:
            print("Warning! Multiple contours!")
        if len(contours) == 0:
            raise Exception("Error! No contour!")

        approx = cv2.approxPolyDP(contours[0], 3.8, True)
        polygon = (approx.squeeze()).tolist()
        return polygon

    @classmethod
    def from_mask_img(cls,
                      mask_img: np.ndarray,
                      is_smoothed: bool = False,
                      cell_id=None):
        """Build a Cell object from a cell mask image.

        Args:
            mask_img (np.darray): A mask img of the cell.
            is_smoothed (str, optional): Whether the mask image is smoothed during interpolation. Defaults to "False".
        Returns:c
            Cell: A cell object.
        """
        if is_smoothed:
            pass  # TODO should sharp the mask img

        # get contour
        bbox = mask_sorted_bbox(mask_img)

        if cell_id is None:
            raise Exception("Please give a cell id!")

        return cls(cell_id,
                   bbox[0].tolist(),
                   bbox[2].tolist(),
                   mask_img.shape,
                   mask=mask_img)


def mask_sorted_bbox(region_mask):
    """Get the sorted bbox of a region mask.

    Args:
        region_mask (np.ndarray): is a uint8 image where pixels in region have 255 value and 0 for the background.

    return:
        List[List[int, int]]: a sorted bbox --> [up_left, up_right, down_right, down_left]
    """
    contour, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL,
                                  cv2.CHAIN_APPROX_SIMPLE)
    if len(contour) > 1:
        print("Warning! Multiple contours!")
    if len(contour) == 0:
        raise Exception("Error! No contour!")
    contour = contour[0]
    bbox = contour_sorted_bbox(contour)
    return bbox


def contour_sorted_bbox(contour):
    """Get the bbox of a contour.
    Args:
        contour (List[List[int, int], ...]):
            [[w1, h1], [w2, h2] ...]
    Returns:
        List[List[int, int]]: a sorted bbox --> [up_left, up_right, down_right, down_left]
    """
    contour = np.array(contour, dtype="float32")
    box_ = cv2.minAreaRect(contour)
    bbox_ = cv2.boxPoints(box_)
    bbox_ = sort_bbox(bbox_)
    return bbox_


def sort_bbox(points: np.ndarray) -> np.ndarray:
    """An unsorted 4-points list of a bbox: [[1, 1], [0, 1], [1, 0], [0, 0]] (in image axis)
       Then the sorted bbox: [[0, 0], [1, 0], [1, 1], [0, 1]] (in image axis)

    Args:
        points (np.ndarray): [description]

    Returns:
        np.ndarray: [description]
    """
    ret = np.array(points).reshape(-1, 2)
    x_add_y = ret[:, 0] + ret[:, 1]
    x_sub_y = ret[:, 0] - ret[:, 1]
    idx0, idx1 = np.argmax(x_sub_y), np.argmin(x_add_y)
    idx2, idx3 = np.argmin(x_sub_y), np.argmax(x_add_y)
    return ret[[idx1, idx0, idx3, idx2]]


class TableBody():
    """A class of table body stores all possible raw info. on original image.
       This class is only for storing. DO NOT implement any heavy model methods in this class !!
    """

    def __init__(self,
                 cells: List,
                 img_shape,
                 origin_img_filename=None,
                 flag_fill_mask_gap=True) -> None:
        """
        Args:
            cells (List): a list of all cells
            flag_analyse (bool, optional): whether to generate structure graph and html string. Defaults to False.
        """
        self.cells: List = cells  # a list of all cells
        self.cell_num = len(cells)  # number of cells
        self.origin_img_filename = origin_img_filename
        self.img_shape = img_shape
        self.flag_fill_mask_gap = flag_fill_mask_gap
        self.mask = self._generate_mask_from_cells_mask(flag_fill_mask_gap)
        self.mask_contour = None
        self.colored_mask = None  # differentiate mask of cells with pixel value of corresponding cell ID. Isn't filled gaps!

    def _generate_mask_from_cells_mask(self, flag_fill_gap):
        """Combine all cell masks to generate mask of table body.

        Args:
            flag_fill_gap (bool, optional): Whether to use morphology methods to fill
            the gaps between cells. Defaults to False.

        Returns:
            Tuple[0] boolean numpy 2-darray: table body mask.
            Tuple[1] uint8 numpy 2-darray: pixel value is corresponding cell id.
        """
        mask = np.zeros(self.img_shape[:2], dtype="bool")
        for cell in self.cells:
            mask = mask | cell.mask
        if flag_fill_gap:
            tmp_mask = mask * 255  # convert boolean matrix to image matrix for image proccessing
            tmp_mask = tmp_mask.astype('uint8')
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
            tmp_mask = cv2.morphologyEx(tmp_mask, cv2.MORPH_CLOSE, kernel)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
            tmp_mask = cv2.dilate(tmp_mask, kernel)
            mask = tmp_mask == 255  # convert back to boolean matrix
        return mask

    def copy_property_from_instance(self,
                                    table_body,
                                    property_="cells_main_properties") -> None:
        if property_ == "cells_main_properties":
            dict_ = {}
            # build dict. for mapping between cells of the two tables
            for cell_ in table_body.cells:
                dict_[cell_.id] = cell_
            for cell_ in self.cells:
                tg_cell_ = dict_[cell_.id]
                cell_.row_index = tg_cell_.row_index
                cell_.colum_index = tg_cell_.colum_index
                cell_.colspan = tg_cell_.colspan
                cell_.rowspan = tg_cell_.rowspan

    def get_mask_contour(self):
        if not self.mask_contour:
            tmp = (self.mask * 255).astype("uint8")
            mask_contour, _ = cv2.findContours(tmp, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)
        if len(mask_contour) > 1:
            print("Warning! Multiple contours!")

        if len(mask_contour) == 0:
            raise Exception("No contour!")

        self.mask_contour = mask_contour[0]
        return mask_contour[0]


def sort_cells(cells) -> List:
    """Sort cells and give the main properties to every cell object.
       This function will empty the list at last!
    Args:
        cells (List[Cell]): a list of cell objects.
    Return:
        sorted cell list.
    """
    result = cells.copy()
    cells_H = min([i.downright[1] - i.upleft[1] for i in result])
    # 规则算法排序单元格
    row_index = -1
    html_res = []
    while (result):
        new_row = []
        flag = True
        state = 1
        row_index += 1
        while (flag):
            if state == 1:
                top_spot = min(result, key=lambda x: x.upleft[1])
                top_spot = result.pop(result.index(top_spot))
                top_spot.row_index = row_index
                new_row.append(top_spot)
                state = 2
            elif state == 2:  # 向右推进
                now_spot = top_spot
                vec = get_vec(result, now_spot, cells_H, 'right', 'upleft')
                while (vec):
                    vec.sort(key=takeSecond)
                    vec = vec[:2 if len(vec) >= 2 else len(vec)]
                    if len(vec) == 2:
                        angle = [abs(vec[0][2]), abs(vec[1][2])]
                        dis_sub = [vec[0][1], vec[1][1]]
                        if abs(vec[0][2] - vec[1][2]) <= 5:
                            first = (vec[0][1] - max(dis_sub)) / max(dis_sub)
                            second = (vec[1][1] - max(dis_sub)) / max(dis_sub)
                        else:
                            try:
                                first = 0.32 * (vec[0][2]) / max(
                                    angle) + 0.68 * (vec[0][1]) / max(dis_sub)
                                second = 0.32 * (vec[1][2]) / max(
                                    angle) + 0.68 * (vec[1][1]) / max(dis_sub)
                            except:
                                first = (vec[0][1] -
                                         max(dis_sub)) / max(dis_sub)
                                second = (vec[1][1] -
                                          max(dis_sub)) / max(dis_sub)
                        index = 0 if first < second else 1
                    else:
                        index = 0
                    result.remove(vec[index][0])
                    vec[index][0].row_index = row_index
                    now_spot = vec[index][0]
                    new_row.append(now_spot)
                    vec = get_vec(result, now_spot, cells_H, 'right', 'upleft')
                # -------------------------------------------------------
                now_spot = top_spot
                vec = get_vec(result, now_spot, cells_H, 'left', 'upleft')
                while (vec):
                    vec.sort(key=takeSecond)
                    vec = vec[:2 if len(vec) >= 2 else len(vec)]
                    if len(vec) == 2:
                        angle = [abs(vec[0][2]), abs(vec[1][2])]
                        dis_sub = [vec[0][1], vec[1][1]]
                        if abs(vec[0][2] - vec[1][2]) <= 5:
                            first = (vec[0][1] - max(dis_sub)) / max(dis_sub)
                            second = (vec[1][1] - max(dis_sub)) / max(dis_sub)
                        else:
                            try:
                                first = 0.32 * (max(angle) - vec[0][2]) / max(
                                    angle) + 0.68 * (vec[0][1]) / max(dis_sub)
                                second = 0.32 * (max(angle) - vec[1][2]) / max(
                                    angle) + 0.68 * (vec[1][1]) / max(dis_sub)
                            except:
                                first = (vec[0][1] -
                                         max(dis_sub)) / max(dis_sub)
                                second = (vec[1][1] -
                                          max(dis_sub)) / max(dis_sub)
                        index = 0 if first < second else 1
                    else:
                        index = 0
                    result.remove(vec[index][0])
                    vec[index][0].row_index = row_index
                    now_spot = vec[index][0]
                    new_row.insert(0, now_spot)
                    vec = get_vec(result, now_spot, cells_H, 'left', 'upleft')
                flag = False
        html_res.append(new_row)

    result = cells
    row_index = -1
    while (result):
        flag = True
        state = 1
        row_index += 1
        while (flag):
            if state == 1:  # 找左上角
                top_spot = min(result, key=lambda x: x.downright[1])
                top_spot = result.pop(result.index(top_spot))
                top_spot.downright_row_index = row_index
                state = 2
            elif state == 2:  # 向右推进
                now_spot = top_spot
                vec = get_vec(result, now_spot, cells_H, 'right', 'downright')
                while (vec):
                    vec.sort(key=takeSecond)
                    vec = vec[:2 if len(vec) >= 2 else len(vec)]
                    if len(vec) == 2:
                        angle = [abs(vec[0][2]), abs(vec[1][2])]
                        dis_sub = [vec[0][1], vec[1][1]]
                        if abs(vec[0][2] - vec[1][2]) <= 5:
                            first = (vec[0][1] - max(dis_sub)) / max(dis_sub)
                            second = (vec[1][1] - max(dis_sub)) / max(dis_sub)
                        else:
                            try:
                                first = 0.32 * (vec[0][2]) / max(
                                    angle) + 0.68 * (vec[0][1]) / max(dis_sub)
                                second = 0.32 * (vec[1][2]) / max(
                                    angle) + 0.68 * (vec[1][1]) / max(dis_sub)
                            except:
                                first = (vec[0][1] -
                                         max(dis_sub)) / max(dis_sub)
                                second = (vec[1][1] -
                                          max(dis_sub)) / max(dis_sub)
                        index = 0 if first < second else 1
                    else:
                        index = 0
                    result.remove(vec[index][0])
                    vec[index][0].downright_row_index = row_index
                    now_spot = vec[index][0]
                    vec = get_vec(result, now_spot, cells_H, 'right',
                                  'downright')
                # -------------------------------------------------------
                now_spot = top_spot
                vec = get_vec(result, now_spot, cells_H, 'left', 'downright')
                while (vec):
                    vec.sort(key=takeSecond)
                    vec = vec[:2 if len(vec) >= 2 else len(vec)]
                    if len(vec) == 2:
                        angle = [abs(vec[0][2]), abs(vec[1][2])]
                        dis_sub = [vec[0][1], vec[1][1]]
                        if abs(vec[0][2] - vec[1][2]) <= 5:
                            first = (vec[0][1] - max(dis_sub)) / max(dis_sub)
                            second = (vec[1][1] - max(dis_sub)) / max(dis_sub)
                        else:
                            try:
                                first = 0.32 * (max(angle) - vec[0][2]) / max(
                                    angle) + 0.68 * (vec[0][1]) / max(dis_sub)
                                second = 0.32 * (max(angle) - vec[1][2]) / max(
                                    angle) + 0.68 * (vec[1][1]) / max(dis_sub)
                            except:
                                first = (vec[0][1] -
                                         max(dis_sub)) / max(dis_sub)
                                second = (vec[1][1] -
                                          max(dis_sub)) / max(dis_sub)
                        index = 0 if first < second else 1
                    else:
                        index = 0
                    result.remove(vec[index][0])
                    vec[index][0].downright_row_index = row_index
                    now_spot = vec[index][0]
                    vec = get_vec(result, now_spot, cells_H, 'left',
                                  'downright')
                flag = False

    # 求span
    Cells_list = sum(html_res, [])
    for cell in Cells_list:
        temp = [i for i in Cells_list if i != cell]
        col_dect = []
        col_span = (cell.upleft[0], 0, cell.downright[0], cell.img_shape[0])
        for other_cell in temp:
            if compute_area_prop(
                    col_span, other_cell.upleft + other_cell.downright) > 0.68:
                col_dect.append(other_cell.row_index)
        cell.colspan = max(
            [col_dect.count(i) for i in set(col_dect)]
            or [1]) if max([col_dect.count(i)
                            for i in set(col_dect)] or [1]) != 0 else 1
        cell.rowspan = cell.downright_row_index - cell.row_index + 1 if cell.downright_row_index - cell.row_index + 1 > 0 else 1

    # add row/colum index
    res_ = []
    for row_id, cell_row in enumerate(html_res):
        for col_id, cell in enumerate(cell_row):
            cell.row_index = row_id + 1
            cell.colum_index = col_id + 1
            res_.append(cell)

    return res_


def takeSecond(elem):
    return elem[1]


def takeThird(elem):
    return elem[2]


def get_vec(result, now_spot, cells_H, position, spot):
    height_th = 0.6
    if spot == 'upleft':
        if position == 'right':
            vec = [
                (i,
                 math.sqrt(
                     pow(i.upleft[1] - now_spot.upleft[1], 2) +
                     pow(i.upleft[0] - now_spot.upleft[0], 2)),
                 math.degrees(
                     math.atan2(i.upleft[1] - now_spot.upleft[1],
                                i.upleft[0] - now_spot.upleft[0])))
                for i in result if math.degrees(
                    math.atan2(i.upleft[1] - now_spot.upleft[1], i.upleft[0] -
                               now_spot.upleft[0])) > -90 and math.degrees(
                    math.atan2(i.upleft[1] -
                               now_spot.upleft[1], i.upleft[0] -
                               now_spot.upleft[0])) <= 18
                                   and abs(i.upleft[1] - now_spot.upleft[1]) < cells_H * height_th
            ]
        elif position == 'left':
            vec = [
                (i,
                 math.sqrt(
                     pow(i.upleft[1] - now_spot.upleft[1], 2) +
                     pow(i.upleft[0] - now_spot.upleft[0], 2)),
                 math.degrees(
                     math.atan2(now_spot.upleft[1] - i.upleft[1],
                                now_spot.upleft[0] - i.upleft[0])))
                for i in result if math.degrees(
                    math.atan2(now_spot.upleft[1] -
                               i.upleft[1], now_spot.upleft[0] -
                               i.upleft[0])) > -90 and math.degrees(
                    math.atan2(now_spot.upleft[1] -
                               i.upleft[1], now_spot.upleft[0] -
                               i.upleft[0])) <= 18
                                   and abs(i.upleft[1] - now_spot.upleft[1]) < cells_H * height_th
            ]
    else:
        if position == 'right':
            vec = [(i,
                    math.sqrt(
                        pow(i.downright[1] - now_spot.downright[1], 2) +
                        pow(i.downright[0] - now_spot.downright[0], 2)),
                    math.degrees(
                        math.atan2(i.downright[1] - now_spot.downright[1],
                                   i.downright[0] - now_spot.downright[0])))
                   for i in result if math.degrees(
                    math.atan2(i.downright[1] -
                               now_spot.downright[1], i.downright[0] -
                               now_spot.downright[0])) > -90
                   and math.degrees(
                    math.atan2(i.downright[1] -
                               now_spot.downright[1], i.downright[0] -
                               now_spot.downright[0])) <= 18
                   and abs(i.downright[1] -
                           now_spot.downright[1]) < cells_H * height_th]
        elif position == 'left':
            vec = [(i,
                    math.sqrt(
                        pow(i.downright[1] - now_spot.downright[1], 2) +
                        pow(i.downright[0] - now_spot.downright[0], 2)),
                    math.degrees(
                        math.atan2(now_spot.downright[1] - i.downright[1],
                                   now_spot.downright[0] - i.downright[0])))
                   for i in result if math.degrees(
                    math.atan2(now_spot.downright[1] -
                               i.downright[1], now_spot.downright[0] -
                               i.downright[0])) > -90
                   and math.degrees(
                    math.atan2(now_spot.downright[1] -
                               i.downright[1], now_spot.downright[0] -
                               i.downright[0])) <= 18
                   and abs(i.downright[1] -
                           now_spot.downright[1]) < cells_H * height_th]
    return vec


def compute_area_prop(rec1, rec2):
    """[summary]

    Args:
        rec1 ([type]): [description]
        rec2 ([type]): [description]

    Returns:
        [type]: [description]
    """
    left_column_max = max(rec1[0], rec2[0])
    right_column_min = min(rec1[2], rec2[2])
    up_row_max = max(rec1[1], rec2[1])
    down_row_min = min(rec1[3], rec2[3])
    # 两矩形无相交区域的情况
    if left_column_max >= right_column_min or down_row_min <= up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
        S2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
        S_cross = (down_row_min - up_row_max) * (right_column_min -
                                                 left_column_max)
        return S_cross / S2


def contours_preprocess(bin_img: np.ndarray):
    img = bin_img.copy()
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))
    img = cv2.dilate(img, kernel)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    return contours, img


cal_d = lambda ptr, theta_radian: ptr[0] * np.sin(theta_radian) + ptr[
    1] * np.cos(theta_radian)


def sort_points(points):
    ret = np.array(points).reshape(-1, 2)
    x_add_y = ret[:, 0] + ret[:, 1]
    x_sub_y = ret[:, 0] - ret[:, 1]
    idx0, idx1 = np.argmax(x_sub_y), np.argmin(x_add_y)
    idx2, idx3 = np.argmin(x_sub_y), np.argmax(x_add_y)
    return ret[[idx1, idx0, idx3, idx2]].reshape(-1).tolist()


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = list(cv2.boxPoints(bounding_box))
    box = sort_points(points)
    return np.array(box).reshape(-1, 2), 90 - bounding_box[2], bounding_box[0]


def draw_line(input_img, theta_d, d, color=(255, 0, 0), thickness=2):
    canvas = input_img.copy()
    h, w, _ = canvas.shape
    ptrs_ls = []
    theta_radian = theta_d * np.pi / 180.0
    if theta_d < 45 or theta_d > 135:
        for x in range(w):
            y = -(x * np.sin(theta_radian) - d) / np.cos(theta_radian)
            if y >= 0 and y < h:
                ptrs_ls.append((int(x), int(y)))
    else:
        for y in range(h):
            x = (-y * np.cos(theta_radian) + d) / np.sin(theta_radian)
            if x >= 0 and x < w:
                ptrs_ls.append((int(x), int(y)))
    if len(ptrs_ls) < 2:
        print("Out of canvas")
        return canvas
    ptr_1, ptr_2 = ptrs_ls[0], ptrs_ls[-1]
    cv2.line(canvas, ptr_1, ptr_2, color, thickness)
    return canvas


def detect_4_ptrs(contour: np.ndarray, input_canvas: np.ndarray = None):
    if input_canvas is not None:
        canvas = input_canvas.copy()
    else:
        canvas = None
    pts, angle, cen_ptr = get_mini_boxes(contour)
    theta1, theta2 = angle - 45, angle + 45
    if theta1 < 0:
        swap_theta = theta1
        theta1 = theta2
        theta2 = swap_theta + 180
    idx_buffer = [-1, -1, -1, -1]
    d1_min, d1_max, d2_min, d2_max = 1e8, -1e8, 1e8, -1e8
    for p_idx, ptr in enumerate(contour):
        temp_theta1 = theta1 * np.pi / 180.0
        temp_theta2 = theta2 * np.pi / 180.0
        temp_x, temp_y = ptr[0], -ptr[1]  # y取-是为了把图像放到第四象限计算
        temp_d1 = temp_x * np.sin(temp_theta1) - temp_y * np.cos(temp_theta1)
        temp_d2 = temp_x * np.sin(temp_theta2) - temp_y * np.cos(temp_theta2)
        if temp_d1 < d1_min:
            d1_min = temp_d1
            idx_buffer[0] = p_idx
        if temp_d1 > d1_max:
            d1_max = temp_d1
            idx_buffer[2] = p_idx
        if temp_d2 < d2_min:
            d2_min = temp_d2
            idx_buffer[3] = p_idx
        if temp_d2 > d2_max:
            d2_max = temp_d2
            idx_buffer[1] = p_idx
    cornor_4_ptrs = contour[idx_buffer]
    if canvas is not None:
        canvas = draw_line(canvas, theta1, d1_min)
        canvas = draw_line(canvas, theta1, d1_max)
        canvas = draw_line(canvas, theta2, d2_min)
        canvas = draw_line(canvas, theta2, d2_max)
        for p_idx, ptr in enumerate(cornor_4_ptrs):
            cv2.circle(canvas,
                       tuple(ptr.tolist()),
                       4, (0, 0, 100 + 50 * p_idx),
                       thickness=7)
    return cornor_4_ptrs, angle, canvas


def gen_target_ptrs(input_cornor_4ptrs, num_ptrs):
    cornor_4ptrs = input_cornor_4ptrs.copy()
    left_ptr, right_ptr = cornor_4ptrs[0], cornor_4ptrs[1]
    ptrs_up_ls = [
        left_ptr * (num_ptrs - 1 - i) / (num_ptrs - 1) + right_ptr * i /
        (num_ptrs - 1) for i in range(num_ptrs)
    ]
    left_ptr, right_ptr = cornor_4ptrs[3], cornor_4ptrs[2]
    ptrs_down_ls = [
        left_ptr * (num_ptrs - 1 - i) / (num_ptrs - 1) + right_ptr * i /
        (num_ptrs - 1) for i in range(num_ptrs)
    ]
    return np.stack(
        [np.stack(ptrs_up_ls, axis=0),
         np.stack(ptrs_down_ls, axis=0)], axis=1)


def cal_line_from_2ptrs(ptr1, ptr2):
    line_vec = ptr1 - ptr2
    line_vec = line_vec / np.sqrt(np.power(line_vec, 2).sum())
    theta_radian = np.arccos(line_vec[0])
    # theta_radian = np.arcsin(line_vec[1])
    # k = line_vec[1] / (line_vec[0] + 1e-10)
    # theta_radian = np.arctan(k)
    if theta_radian < 0:
        theta_radian += np.pi
    d = cal_d(ptr1, theta_radian)
    return theta_radian, d


def cal_is_ptrs(line_ls, contour, img_width, img_height):
    mask_lines = np.zeros((img_height, img_width, 3)).astype(np.uint8)
    maks_contour = np.zeros((img_height, img_width, 3)).astype(np.uint8)
    for line in line_ls:
        theta_d = line[0] * 180 / np.pi
        d = line[1]
        mask_lines = draw_line(mask_lines,
                               theta_d,
                               d,
                               color=(255, 255, 255),
                               thickness=3)
    contour = np.expand_dims(contour, axis=1)
    cv2.drawContours(maks_contour, [contour], -1, (255, 255, 255), 3)
    dst = cv2.bitwise_and(maks_contour, mask_lines)
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    dst = cv2.dilate(dst, None)
    cnts, _ = cv2.findContours(dst, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    ptrs = []
    for cnt in cnts:
        temp_moment = cv2.moments(cnt)
        x_c = temp_moment["m10"] / temp_moment["m00"]
        y_c = temp_moment["m01"] / temp_moment["m00"]
        ptrs.append([x_c, y_c])
    ptrs = np.array(ptrs)
    if ptrs.shape[0] != 2 * len(line_ls):
        return None
    d_mat = np.zeros((len(ptrs), len(line_ls)))
    for row in range(d_mat.shape[0]):
        for col in range(d_mat.shape[1]):
            d_mat[row, col] = abs(
                cal_d(ptrs[row], line_ls[col][0]) - line_ls[col][1])
    ret = np.zeros((len(line_ls), 2, 2))
    idx_ls = np.argmin(d_mat, axis=1)
    for line_idx in range(len(line_ls)):
        temp_ptrs_pair = ptrs[idx_ls == line_idx]
        if len(temp_ptrs_pair) != 2:
            return None
        if temp_ptrs_pair[0, 1] > temp_ptrs_pair[1, 1]:
            swap_ptr = temp_ptrs_pair[0].copy()
            temp_ptrs_pair[0] = temp_ptrs_pair[1]
            temp_ptrs_pair[1] = swap_ptr
        ret[line_idx] = temp_ptrs_pair
    return ret


def get_rotate_crop_image(img,
                          points,
                          interp_mode=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE):  # INTER_NEAREST
    img_crop_width = int(
        max(np.linalg.norm(points[0] - points[1]),
            np.linalg.norm(points[2] - points[3])))
    img_crop_height = int(
        max(np.linalg.norm(points[0] - points[3]),
            np.linalg.norm(points[1] - points[2])))
    pts_std = np.float32([
        [0, 0],
        [img_crop_width, 0],
        [img_crop_width, img_crop_height],
        [0, img_crop_height],
    ])
    M = cv2.getPerspectiveTransform(points.astype(np.float32), pts_std)

    dst_img = cv2.warpPerspective(
        img,
        M,
        (img_crop_width, img_crop_height),
        borderMode=borderMode,
        flags=interp_mode,
    )
    return dst_img


def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)
    # original implementation, very slow
    # pairwise_dist = torch.sum(pairwise_diff ** 2, dim = 2) # square of distance
    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)
    # fix numerical error for 0 * log(0), substitute all nan with 0
    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


class TPSGridGen(nn.Module):
    def __init__(self, target_height, target_width, target_control_points):
        super(TPSGridGen, self).__init__()
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)

        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
        Y, X = target_coordinate.split(1, dim=1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim=1)  # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim=1)

        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate


def grid_sample(input, grid, canvas=None, interp_mode="bilinear"):
    output = F.grid_sample(input, grid, mode=interp_mode)
    if canvas is None:
        return output
    else:
        input_mask = input.data.new(input.size()).fill_(1)
        output_mask = F.grid_sample(input_mask, grid, mode=interp_mode)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output


class TPS:
    def __init__(self,
                 source_ptrs: np.ndarray,
                 target_ptrs: np.ndarray,
                 img_width,
                 img_height,
                 interp_mode="bilinear") -> None:
        self.img_width, self.img_height = img_width, img_height
        self.src_ptrs = source_ptrs.copy().reshape(-1, 2).astype(np.float32)
        self.tar_ptrs = target_ptrs.copy().reshape(-1, 2).astype(np.float32)
        self.src_ptrs[:, 0] = self.src_ptrs[:, 0] / img_width
        self.src_ptrs[:, 1] = self.src_ptrs[:, 1] / img_height
        self.tar_ptrs[:, 0] = self.tar_ptrs[:, 0] / img_width
        self.tar_ptrs[:, 1] = self.tar_ptrs[:, 1] / img_height
        self.src_ptrs = torch.from_numpy(self.src_ptrs * 2 - 1)
        self.tar_ptrs = torch.from_numpy(self.tar_ptrs * 2 - 1)
        self.tps = TPSGridGen(self.img_height, self.img_width, self.tar_ptrs)
        source_coordinate = self.tps(torch.unsqueeze(self.src_ptrs, 0))
        self.grid = source_coordinate.view(1, self.img_height, self.img_width,
                                           2)
        self.interp_mode = interp_mode

    def __call__(self, input_img):
        img = input_img.copy().astype("float32")
        img = np.expand_dims(np.transpose(img, (2, 0, 1)), 0)
        img = torch.from_numpy(img)
        canvas = torch.Tensor(1, 3, self.img_height, self.img_width).fill_(0)  # 255
        target_image = grid_sample(img,
                                   self.grid,
                                   canvas,
                                   interp_mode=self.interp_mode)
        target_image = (
            target_image.detach().cpu().numpy().squeeze().transpose(
                1, 2, 0).astype(np.uint8))
        return target_image


class Corrector:
    def __init__(self,
                 contour,
                 img_width,
                 img_height,
                 num_ctl_ptrs_pair=5,
                 show_flag=False,
                 canvas=None,
                 img_type="pixelImg") -> None:
        self.disable_flag = False
        self.show_flag = show_flag
        if self.show_flag:
            if canvas is not None:
                self.canvas = canvas.copy()
            else:
                self.canvas = np.zeros(size=(img_height, img_width, 3))
        else:
            self.canvas = None
        self.img_height, self.img_width = img_height, img_width
        self.contour = contour
        if self.show_flag:
            self._draw_contour()
        self.contour = self.contour.squeeze(1)
        self.roi_range = {
            "x_min": self.contour[:, 0].min(),
            "y_min": self.contour[:, 1].min(),
            "width": self.contour[:, 0].max() + 1 - self.contour[:, 0].min(),
            "height": self.contour[:, 1].max() + 1 - self.contour[:, 1].min(),
        }
        self.cornor_ptrs, self.angle, self.canvas = detect_4_ptrs(
            self.contour, self.canvas)
        self.angle_radian = self.angle * np.pi / 180.0
        self.target_ptrs = gen_target_ptrs(self.cornor_ptrs, num_ctl_ptrs_pair)
        self.line_ls = [
            cal_line_from_2ptrs(temp_ptr_pair[0], temp_ptr_pair[1])
            for temp_ptr_pair in self.target_ptrs[1:-1]
        ]
        self.source_ptrs = self.target_ptrs.copy()
        is_ptrs = cal_is_ptrs(self.line_ls, self.contour, self.img_width,
                              self.img_height)
        if is_ptrs is None:
            self.disable_flag = True
            return
        self.source_ptrs[1:-1] = is_ptrs

        self.lt_ptr = np.array(
            [self.roi_range["x_min"], self.roi_range["y_min"]])
        self.src_ptrs_roi = self.source_ptrs - self.lt_ptr
        self.tar_ptrs_roi = self.target_ptrs - self.lt_ptr
        self.cor_ptrs_roi = self.cornor_ptrs - self.lt_ptr

        self.img_type = img_type

        if img_type == "pixelImg":
            self.tps_interp_mode = "bilinear"
            self.proj_interp_mode = cv2.INTER_CUBIC
            self.borderMode = cv2.BORDER_REPLICATE
        elif img_type == "idImg":
            self.tps_interp_mode = "nearest"
            self.proj_interp_mode = cv2.INTER_NEAREST
            self.borderMode = cv2.BORDER_REPLICATE
        else:
            raise Exception("Please give a correct image type!")

        self.tps = TPS(self.src_ptrs_roi, self.tar_ptrs_roi,
                       self.roi_range["width"], self.roi_range["height"],
                       interp_mode=self.tps_interp_mode)

        if self.show_flag:
            self._draw_line_ls()
            self._draw_target_ptrs()
            self._draw_source_ptrs()
            # cv2.imwrite("canvas.jpg", self.canvas)

    def __call__(self, input_img):
        img = input_img.copy()
        roi_img = img[self.roi_range["y_min"]:self.roi_range["y_min"] +
                                              self.roi_range["height"],
                  self.roi_range["x_min"]:self.roi_range["x_min"] +
                                          self.roi_range["width"], ]
        if self.disable_flag:
            return roi_img
        out_img = self.tps(roi_img)
        out_img = get_rotate_crop_image(out_img,
                                        self.cor_ptrs_roi,
                                        interp_mode=self.proj_interp_mode,
                                        borderMode=self.borderMode)
        return out_img

    def _draw_target_ptrs(self):
        temp_ptrs = self.target_ptrs.reshape(-1, 2)
        for ptr in temp_ptrs:
            ptr = (int(ptr[0]), int(ptr[1]))
            cv2.circle(self.canvas, ptr, 4, (255, 255, 0), 2)

    def _draw_line_ls(self):
        for theta, d in self.line_ls:
            theta = theta * 180.0 / np.pi
            self.canvas = draw_line(self.canvas, theta, d)

    def _draw_source_ptrs(self):
        temp_ptrs = self.source_ptrs.reshape(-1, 2)
        for ptr in temp_ptrs:
            ptr = (int(ptr[0]), int(ptr[1]))
            cv2.circle(self.canvas, ptr, 2, (0, 255, 255), 3)

    def _draw_contour(self):
        self.canvas = cv2.drawContours(self.canvas, [self.contour],
                                       -1, (0, 255, 0),
                                       thickness=2)


class Tables():
    def __init__(self, tables: List):
        """Initialize instance from tableBody instance and group id tuple list

        Args:
            tables (List): tableBody instance and group id tuple list, e.g. [(tb1, 0), (tb2, 1), ...]
        """
        self.tables: List = tables
        if self.tables:
            self.origin_img_filename = self.tables[0][0].origin_img_filename
            self.img_shape = self.tables[0][0].img_shape

    def generate_labelme_json(self, save_path: str = None, labelme_checked_flag=False) -> str:
        temp = {
            "version": "4.5.9",
            "flags": {},
            "shapes": [],
            "lineColor": [0, 255, 0, 128],
            "fillColor": [255, 0, 0, 128],
            "imagePath": self.origin_img_filename,
            "imageData": None,
            "imageHeight": self.img_shape[0],
            "imageWidth": self.img_shape[1],
            "checked": labelme_checked_flag
        }

        for tb_tuple in self.tables:
            cells = tb_tuple[0].cells
            group_id = tb_tuple[1]
            # generate polygons field  w or w/o structure or content
            for index, cell in enumerate(cells):
                if cell.has_main_property():
                    row_id = cell.row_index
                    colum_id = cell.colum_index
                    rowspan = cell.rowspan
                    colspan = cell.colspan
                    content = cell.content
                    label: str = "{0}-{1}-{2}-{3}-".format(
                        row_id, colum_id, rowspan, colspan)
                    # Discriminate difference between empty content and the no OCR results
                    if content is not None:
                        label += "{}".format(content)
                else:
                    label = None
                shapes_temp = {
                    "label": label,
                    "line_color": None,
                    "fill_color": [255, 0, 0, 0],
                    "points": [],
                    "shape_type": "polygon",
                    "group_id": group_id
                }
                shapes_temp["points"] = cell.polygon
                temp["shapes"].append(shapes_temp)

        if save_path is not None:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(temp, f, ensure_ascii=False, indent=4)

        return temp


def main(json_path, output_path):
    # load json
    with open(json_path, encoding='utf-8') as file:
        labelme_annos = json.load(file)

    # initialize parameters
    polygons = labelme_annos["shapes"]
    img_filename = labelme_annos["imagePath"]
    image_shape = labelme_annos["imageHeight"], labelme_annos["imageWidth"]

    polygons_groupBy_groupID = {}
    for poly_ in polygons:
        key_ = poly_["group_id"]
        if polygons_groupBy_groupID.get(key_) is None:
            polygons_groupBy_groupID[key_] = []
        polygons_groupBy_groupID[key_].append(poly_)

    tb_gid_tuple_list = []
    for group_id, polygons in polygons_groupBy_groupID.items():
        # load cells and initialize TableBody object
        cells: List[Cell] = []
        for idx_, polygon_ in enumerate(polygons):
            cell_ = Cell(idx_,
                         None,
                         None,
                         image_shape,
                         polygon=polygon_["points"])
            cells.append(cell_)
        table_body = TableBody(cells,
                               cells[0].img_shape,
                               origin_img_filename=img_filename)

        # get contour of table body for Corrector 
        contour = table_body.get_mask_contour()

        img_height, img_width = table_body.img_shape[:2]

        # initialize Corrector object
        corrector_idImg = Corrector(contour,
                                    img_width,
                                    img_height,
                                    num_ctl_ptrs_pair=7,
                                    img_type="idImg")

        # sort cells to get table structure attributes 
        dewarped_cells: List = []
        # TODO can only dewarp once on table color mask iff. color mask is sharp not smoothed
        for cell_ in table_body.cells:
            cell_mask_img_ = cell_.get_mask_img()
            cell_mask_img_ = cv2.cvtColor(
                cell_mask_img_, cv2.COLOR_GRAY2RGB
            )  # TODO now corrector only receives 3-channels pic.
            dewarped_cell_mask_img_ = corrector_idImg(cell_mask_img_)
            cell_mask_img_for_build_ = cv2.cvtColor(
                dewarped_cell_mask_img_, cv2.COLOR_RGB2GRAY)
            dewarped_cell_ = Cell.from_mask_img(cell_mask_img_for_build_,
                                                cell_id=cell_.id)
            dewarped_cells.append(dewarped_cell_)

        dewarped_cells = sort_cells(dewarped_cells)

        # create a dewarped TableBody object for properties conveying 
        dewarped_table_body = TableBody(dewarped_cells,
                                        dewarped_cells[0].img_shape)

        # copy structure attributes to original TableBody object
        table_body.copy_property_from_instance(dewarped_table_body)

        tb_gid_tuple_list.append((table_body, group_id))

    tables = Tables(tb_gid_tuple_list)

    # generate json files
    save_path = os.path.join(output_path, os.path.basename(json_path))
    tables.generate_labelme_json(save_path=save_path, labelme_checked_flag=labelme_annos["checked"])


if __name__ == "__main__":
    ######################### inputs #######################
    raw_imgs_path = "X:/LAB/CurveTabSet/test/"
    error_save_path = "X:/LAB/CurveTabSet/errors.txt"
    #multi_tables_txt_path = 'data/refined_CurveTabSet/multi-tables.txt'

    ######################### outputs #######################
    output_path = "X:/LAB/CurveTabSet/test_mask/"
    # whether the file to be written exists

    if os.path.exists(output_path):
        flag_delete_ = input("\033[1;31m%s Exists!! Delete first? yes|no: \033[0m" %
                             os.path.basename(output_path))
        if flag_delete_ == "yes":
            shutil.rmtree(output_path)
        else:
            raise Exception("%s Exists!!" % os.path.basename(output_path))
    os.makedirs(output_path)

    ######################### prepare #######################
    # create multi-table image list
    # with open(multi_tables_txt_path, 'r') as f:
    #     tmp = f.readlines()
    # multi_table_filenames = []
    # for el_ in tmp:
    #     multi_table_filenames.append(
    #         change_file_format(
    #             add_filename_suffix(el_.replace("\n", ""), "_segR_"),
    #             ".json"))
    # del tmp

    # load files
    file_paths = glob.glob(
        os.path.join(raw_imgs_path, "*.json"))

    ######################### do the job #######################
    errors = []
    # main loop
    for path_ in tqdm(file_paths, desc="Generating Annos."):
        try:
            main(path_, output_path)
        except Exception as e:
            trace_back = traceback.format_exc()
            print(trace_back)
            errors.append((os.path.basename(path_), trace_back, e))

    ######################### save errors #######################
    if os.path.exists(error_save_path):
        flag_delete_ = input("\033[1;31m%s Exists!! Delete first? yes|no: \033[0m" %
                             os.path.basename(error_save_path))
        if flag_delete_ == "yes":
            os.remove(error_save_path)
        else:
            raise Exception("%s Exists!!" % os.path.basename(error_save_path))
    with open(error_save_path, 'w') as f:
        for idx_, error in enumerate(errors):
            error_filename = error[0]
            error_trace_back_lines = error[1]
            f.write("NO.{0}, Filename: {1}************************\n".format(
                idx_ + 1, error_filename))
            error_trace_back_lines = error_trace_back_lines.split("\n")
            for trace_back_line_ in error_trace_back_lines:
                f.write(trace_back_line_ + "\n")
            f.write("\n")
    #############################################################

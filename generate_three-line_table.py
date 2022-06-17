"""
    Generate three-line table based on refined labelme
    cell segmentation annotations as well as original image and save to jpg files.

    should input: raw_imgs_path(location of json files and original images)
                  error_save_path(location of error.txt)
                  output_path(save three-line table jpg files in this path)
"""
import os
import traceback
import shutil
import json
import glob, cv2
from tqdm import tqdm
import numpy as np
import math
from typing import List, Tuple


class Cell():
    """A class of table cell stores all possible raw info. on original image.
    """

    def __init__(self,
                 id,
                 img_shape,
                 mask: np.ndarray = None,
                 polygon=None,
                 rowspan=None,
                 colspan=None,
                 row_index=None,
                 colum_index=None):
        """
        Args:
            id (Integer): differentiate and identify each cell
            img_shape (Tuple): shape of original image (Height, Width)
            mask (np.ndarray of bool): cell region mask on original image
            polygon (List[List]): sorted [[w1, h1], [w2, h2] ...]
        """
        self.id = id
        self.colspan = colspan
        self.rowspan = rowspan
        self.row_index = row_index
        self.colum_index = colum_index
        self.img_shape = img_shape

        if mask is None:
            if polygon is not None:
                self.mask: np.ndarray = self._mask_from_polygon(polygon)
            else:
                raise Exception("No enough info. for constructing Cell object!")
        else:
            self.mask: np.ndarray = mask

        if polygon is None:
            if mask is not None:
                self.polygon = self._polygon_from_mask(mask)
            else:
                raise Exception("No enough info. for constructing Cell object!")
        else:
            self.polygon = polygon

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


class WarpRectContour():
    """A rectangle-like closed contour with zero width. Edges may be curve.
    """

    def __init__(self, points, left_edge, up_edge, right_edge, bottom_edge) -> None:
        """
        Args:
            points (List): sorted point list in pixel coordinate which refers to the contour
            left_edge (List): left edge of the contour. pixel coordinate
            up_edge (List): up edge of the contour. pixel coordinate
            right_edge (List): right edge of the contour. pixel coordinate
            bottom_edge (List): bottom edge of the contour. pixel coordinate
        """
        self.points = points  # sorted points

        # all edges are sub set of points
        self.left_edge = left_edge
        self.up_edge = up_edge
        self.right_edge = right_edge
        self.bottom_edge = bottom_edge

    @classmethod
    def from_mask(cls, mask: np.ndarray):
        """
        Args:
            mask (np.ndarray of bools): object contour

        Returns:
            A Contour instance.
        """
        assert mask.dtype == np.dtype("bool")
        mask_ = mask * 255
        mask_ = mask_.astype("uint8")
        mask_[mask_ >= 127] = 255
        mask_[mask_ < 127] = 0
        contours, _ = cv2.findContours(mask_, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        edge_dict, points_list, ptrs = divide_four_edges(contours[0], 4, 3)
        return cls(points_list, edge_dict["left"], edge_dict["top"],
                   edge_dict["right"], edge_dict["bottom"])


def get_mini_boxes(contour):
    bounding_box = cv2.minAreaRect(contour)
    points = list(cv2.boxPoints(bounding_box))
    box = sort_points(points)
    return np.array(box).reshape(-1, 2), 90 - bounding_box[2], bounding_box[0]


def detect_4_ptrs(contour: np.ndarray):
    pts, angle, cen_ptr = get_mini_boxes(contour)
    ################################################
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
    return cornor_4_ptrs, angle


def get_centerpoint(points: list):
    '''
    获取大致重心位置
    （不要使用网上的多边形重心程序，这个程序虽然准确，但对于出现x型自交叉的多边形的重心计算会出现错误，就是假设正确的顺时针顺序是1，2，3，4，实际标注顺序是1，3，2，4时计算的重心会偏出图像外）
    :param points: 点坐标列表；eg:[[x1, y1], [x2, y2], [x3, y3]...]
    :return: x, y: 重心点坐标
    '''
    x, y = 0, 0
    for p in points:
        x += p[0]
        y += p[1]
    return x / len(points), y / len(points)


def sort_points(points):
    '''
    以左上角为起点，顺时针排列points
    :param points: 点坐标列表；eg:[[x1, y1], [x2, y2], [x3, y3]...]
    :return: 排好顺序的点坐标列表；eg:[[x1, y1], [x2, y2], [x3, y3]...]  -> 转成numpy形式
    '''
    center_x, center_y = get_centerpoint(points)
    temp_list = []
    for p in points:
        dy = -(p[1] - center_y)  # 此处因为图片的y轴与通常的xy坐标轴反向，因此取负
        dx = p[0] - center_x
        angle = math.degrees(math.atan2(dy, dx))  # 得到向量夹角
        temp_list.append([p, angle])  # 将夹角与点坐标放入temp_list
    temp_list.sort(key=lambda x: x[1],
                   reverse=True)  # temp_list以夹角由大到小排序，即点坐标以左上角起始，顺时针分布
    return np.asarray([t[0] for t in temp_list])


def divide_four_edges(contour: list, downwardOffset, upwardOffset) -> dict:
    '''
    点排序法划分
        param:
            contour: [ [x1,y1], [x2,y2] ... ]. x: matrix row, y: matrix column
            downwardOffset:向下偏移量；eg: 4
            upwardOffset:向上偏移量 eg:3
        return:
            edge_dict

    '''
    #### 对shape: (length, 1, 2), 降维并调换横纵坐标
    contour = contour.squeeze()[:, (1, 0)]

    ########点排序法
    ptrs, _ = detect_4_ptrs(contour)
    points = sort_points(contour)
    points_list = points.tolist()
    a, b, c, d = points_list.index(ptrs[0].tolist()), points_list.index(
        ptrs[1].tolist()), points_list.index(
        ptrs[2].tolist()), points_list.index(ptrs[3].tolist())
    edge_left = points[(a + downwardOffset):(b - upwardOffset), :]
    edge_bottom = points[b:c, :]
    edge_right = points[(c + downwardOffset):(d - upwardOffset), :]
    edge_top = np.concatenate((points[d:points.shape[0], :], points[0:a, :]),
                              axis=0)
    edge_dict = dict({
        "left": edge_left,
        "top": edge_top,
        "right": edge_right,
        "bottom": edge_bottom
    })
    return edge_dict, points_list, ptrs


def erase_table_edge(img: np.ndarray, points: list, erase_width=8) -> np.ndarray:
    '''
    在原图中根据points列表，处理像素点(中值滤波: kernal_size(default):4)
        param:
            img: 原图
            points: 需处理的像素点的list
        output:
            new_img
    '''
    img = img.copy()
    kernel_size = erase_width  #####################修改################
    height, width, _ = img.shape

    ####形态学处理
    mask = np.zeros((height, width), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for point in points:
        mask[max(0, point[0] - 1):min(height, point[0] + 1),
        max(0, point[1] - 1):min(width, point[1] + 1)] = 1
    ####CROSS操作####
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=2)
    points = np.argwhere(mask == 1)

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    R_tmp, G_tmp, B_tmp = R, G, B
    for p in points:
        [x, y] = p
        R_part = list(
            R_tmp[max(0, x -
                      kernel_size // 2):min(height, x + kernel_size // 2 +
                                            1),
            max(0, y -
                kernel_size // 2):min(width, y + kernel_size // 2 +
                                      1)].reshape(-1))
        G_part = list(
            G_tmp[max(0, x -
                      kernel_size // 2):min(height, x + kernel_size // 2 +
                                            1),
            max(0, y -
                kernel_size // 2):min(width, y + kernel_size // 2 +
                                      1)].reshape(-1))
        B_part = list(
            B_tmp[max(0, x -
                      kernel_size // 2):min(height, x + kernel_size // 2 +
                                            1),
            max(0, y -
                kernel_size // 2):min(width, y + kernel_size // 2 +
                                      1)].reshape(-1))

        R_part.sort(reverse=True)
        G_part.sort(reverse=True)
        B_part.sort(reverse=True)

        ####mid####
        R[x, y] = R_part[len(R_part) * 3 // 7]
        G[x, y] = G_part[len(R_part) * 3 // 7]
        B[x, y] = B_part[len(R_part) * 3 // 7]

    new_img = np.concatenate(
        [np.expand_dims(R, 2),
         np.expand_dims(G, 2),
         np.expand_dims(B, 2)],
        axis=2)

    return new_img


def main(json_path, output_path):
    # get name and load picture
    f_name, ext = os.path.splitext(json_path)
    base_name = os.path.basename(f_name)
    jpg_path = f_name + ".jpg"
    img = cv2.imread(jpg_path)

    # load json
    with open(json_path, encoding='utf-8') as file:
        labelme_annos = json.load(file)

    # initialize parameters
    polygons = labelme_annos["shapes"]
    image_shape = labelme_annos["imageHeight"], labelme_annos["imageWidth"]

    polygons_groupBy_groupID = {}
    for poly_ in polygons:
        key_ = poly_["group_id"]
        if polygons_groupBy_groupID.get(key_) is None:
            polygons_groupBy_groupID[key_] = []
        polygons_groupBy_groupID[key_].append(poly_)

    for group_id, polygons in polygons_groupBy_groupID.items():
        # load cells and initialize TableBody object
        cells: List[Cell] = []
        rows = 1  # 表格总行数
        for idx_, polygon_ in enumerate(polygons):
            cell_ = Cell(idx_,
                         image_shape,
                         polygon=polygon_["points"],
                         row_index=int(polygon_["label"].split('-')[0]),
                         colum_index=int(polygon_["label"].split('-')[1]),
                         rowspan=int(polygon_["label"].split('-')[2]),
                         colspan=int(polygon_["label"].split('-')[3]),
                         )
            if cell_.row_index > rows:
                rows = cell_.row_index  # 计算总行数
            cells.append(cell_)

        if rows < 3:
            for cell in cells:
                contour = WarpRectContour.from_mask(cell.mask)
                img = erase_table_edge(img, contour.right_edge)
                img = erase_table_edge(img, contour.left_edge)
        if rows >= 3:
            for cell in cells:
                contour = WarpRectContour.from_mask(cell.mask)
                if cell.row_index + cell.rowspan - 1 == rows:
                    img = erase_table_edge(img, contour.right_edge)
                    img = erase_table_edge(img, contour.left_edge)
                    img = erase_table_edge(img, contour.up_edge)
                elif cell.row_index == 1:
                    img = erase_table_edge(img, contour.right_edge)
                    img = erase_table_edge(img, contour.left_edge)
                elif cell.row_index == 2:
                    img = erase_table_edge(img, contour.right_edge)
                    img = erase_table_edge(img, contour.left_edge)
                    img = erase_table_edge(img, contour.bottom_edge)
                elif cell.row_index == rows:
                    img = erase_table_edge(img, contour.right_edge)
                    img = erase_table_edge(img, contour.left_edge)
                    img = erase_table_edge(img, contour.up_edge)
                else:
                    img = erase_table_edge(img, contour.right_edge)
                    img = erase_table_edge(img, contour.left_edge)
                    img = erase_table_edge(img, contour.up_edge)
                    img = erase_table_edge(img, contour.bottom_edge)

        cv2.imwrite(output_path + base_name + ".jpg", img)  # 存储图片


if __name__ == "__main__":
    ######################### inputs #######################
    raw_imgs_path = "X:/LAB/CurveTabSet/test/"
    error_save_path = "X:/LAB/CurveTabSet/errors.txt"

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

    ######################### do the job #######################
    file_paths = glob.glob(
        os.path.join(raw_imgs_path, "*.json"))
    errors = []
    # main loop
    for path_ in tqdm(file_paths, desc="Generating three-line table."):
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

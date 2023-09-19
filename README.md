# **TabRecSet: A Large Scale Dataset for End-to-end Table Recognition in the Wild**
Table recognition (TR) is one of the research hotspots in pattern recognition, which aims to extract information from tables in an image. Common table recognition tasks include table detection (TD), table structure recognition (TSR) and table content recognition (TCR). TD is to locate tables in the image, TCR recognizes text content, and TSR recognizes spatial & ontology (logical) structure. Currently, the end-to-end TR in real scenarios, accomplishing the three sub-tasks simultaneously, is yet an unexplored research area. One major factor that inhibits researchers is the lack of a benchmark dataset. To this end, we propose a new large-scale dataset named Table Recognition Set (TabRecSet) with diverse table forms sourcing from multiple scenarios in the wild, providing complete annotation dedicated to end-to-end TR research. It is the largest and first bi-lingual dataset for end-to-end TR, with 38.1 K tables in which 20.4 K are in English and 17.7 K are in Chinese. The samples have diverse forms, such as the border-complete and -incomplete table, regular and irregular table (rotated, distorted, etc.). The scenarios are multiple in the wild, varying from scanned to camera-taken images, documents to Excel tables, educational test papers to financial invoices. The annotations are complete, consisting of the table body spatial annotation, cell spatial & logical annotation and text content for TD, TSR and TCR, respectively. The spatial annotation utilizes the polygon instead of the bounding box or quadrilateral adopted by most datasets. The polygon spatial annotation is more suitable for irregular tables that are common in wild scenarios. Additionally, we propose a visualized and interactive annotation tool named TableMe to improve the efficiency and quality of table annotation.

# Overall Size:
In the aspect of overall size, TabRecSet contains 32,072 images and 38,177 tables in total among which 16,530 images (17,762 tables) are in Chinese, 15,542 images (20,415 tables) are in English and 21,228 images (25,279 tables) are generated (three-line and no-line). The generated table subsets contains 5,113 images and 6728 tables (both three- and no-line tables) in English, 5,501 images and 5,911 tables in Chinese (both three- and no-line tables).

# Samples Overview:
![image](https://user-images.githubusercontent.com/33459391/222026545-070cb416-dd37-4959-b7ea-4af3e099671e.png)

# Comparison With Other Datasets
![image](https://user-images.githubusercontent.com/33459391/222026643-fdda085c-b69c-4037-8d92-36fd59dd56f4.png)

# Download TabRecSet

DOI: 

```
https://doi.org/10.6084/m9.figshare.20647788
```

# The annotation tool TableMe:
Download link: https://doi.org/10.6084/m9.figshare.20647788
![image](https://user-images.githubusercontent.com/33459391/222026699-b7dc0824-3702-464c-8a16-d89db35a6d47.png)

**Demonstration**

![](https://github.com/MaxKinny/TabRecSet/blob/main/demo_TableMe.gif)

**Dependencies**:
Anaconda, PyQT5, Labelme (Open at least one time)

# Please Cite Our Paper
DOI:
```
https://doi.org/10.1038/s41597-023-01985-8
```

citation:
```
@article{yang2023large,
  title={A large-scale dataset for end-to-end table recognition in the wild},
  author={Yang, Fan and Hu, Lei and Liu, Xinwu and Huang, Shuangping and Gu, Zhenghui},
  journal={Scientific Data},
  volume={10},
  number={1},
  pages={110},
  year={2023},
  publisher={Nature Publishing Group UK London}
}
```
or
```
Yang, F., Hu, L., Liu, X. et al. A large-scale dataset for end-to-end table recognition in the wild. Sci Data 10, 110 (2023). https://doi.org/10.1038/s41597-023-01985-8
```

# OCR Annotations
We also provide OCR annotations from Tencent OCR APIs:
![image](https://user-images.githubusercontent.com/33459391/222130324-1f2b5b02-a088-44ab-b855-02f443b7f1ab.png)

![2e4821601e461120d7e762b74e9a892](https://user-images.githubusercontent.com/33459391/222127521-1e38416b-b8d8-4345-a9cb-6c28234ed90a.jpg)

# License:
<a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-sa/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-sa/4.0/">Creative Commons Attribution-ShareAlike 4.0 International License</a>.

# Updates:

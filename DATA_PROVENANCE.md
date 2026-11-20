# Data provenance

The two DGA collections used in the manuscript ship, unmodified, in two archives that the authors received together with the source dissertations. The three spreadsheets in `Data/` are **byte-identical copies** of the files inside those archives: the SHA-256 values below match the SHA-256 values of the corresponding members of the archives.

## The two archives

| archive | size (MB) | SHA-256 | contents |
| --- | --- | --- | --- |
| 变压器故障数据，6特征，带代码带论文数据来源.zip | 26.67 | `076bce786f01caf4111d4f9d67aa884c70e0cf1393cc386dfa5f3365a5acebe7` | expanded collection: 6-feature dataset, MATLAB reduction code and two dissertations |
| DGA原始数据集+扩维数据集.zip | 19.49 | `2c9c5765822fa1f1121f94e3762315fb531b905f70fac4e2848325d830cc1c82` | source collection: raw 570-record file and the 550-record expanded matrix, plus the doctoral dissertation |

## Data files inside the archives, and where they appear in this package

| file inside the archive | size (MB) | SHA-256 | copy in this package |
| --- | --- | --- | --- |
| 变压器溶解气体数据集.xlsx | 0.028 | `32e5097c5c47238a39a1331209ca3d06b08e8131c34cdeb6ece0932868f8ded1` | Data/external_504.xlsx |
| 扩维度DGA电力变压器数据.xlsx | 0.156 | `04f9b92cac2327369061f5833f83b4ec08d505afdb25868781a0fc89a1c5887d` | Data/expanded_dga_550.xlsx |
| 原始.xlsx | 0.031 | `711913b9e581bb589760402c31a657666b4ac1c0cb98cd023e67e01d040f0524` | Data/original_570.xlsx |

## The dissertations the data come from

* Yin, J. *Study on Oil-immersed Power Transformer Fault Diagnosis Based on Relevance Vector Machine*. PhD dissertation, North China Electric Power University, Beijing, 2013, 136 pp. (in Chinese: 基于相关向量机的油浸式电力变压器故障诊断方法研究, 华北电力大学博士学位论文). Source of the 570-record collection and of the 550-record expanded matrix.
* Mei, T. *Research on Power Transformer Fault Diagnosis Method Based on Improved ISOMAP and WkNN*. Master’s thesis, Xihua University, Chengdu, April 2020, 81 pp. (in Chinese: 基于改进 ISOMAP 和 WkNN 的电力变压器故障诊断方法研究, 西华大学硕士学位论文, supervisor Prof. Zhang Bide). Source of the 504-record six-feature collection.

The two dissertations are included in `Source_Documents/` under neutral file names, byte-identical to the PDFs inside the archives. Copyright remains with their authors; the files are reproduced here only so that the data provenance can be verified without access to the original download channels. Their SHA-256 values are listed below.

| dissertation file | size (MB) | SHA-256 |
| --- | --- | --- |
| 基于改进ISOMAP和Wk...电力变压器故障诊断方法研究_梅婷.pdf | 7.33 | `a9d6feef3781c24094989af063840c40f8e5b70b5d8ca4dc5dd377408f2f53ee` |
| 基于相关向量机的油浸式电力变压器故障诊断方法研究_尹金良(1)华电博士论文.pdf | 19.36 | `422f2ee116262c5b22a1d2a20fe8dc2548e1e96d0c51d357002e5ee1d3b1e9e2` |
| 基于相关向量机的油浸式电力变压器故障诊断方法研究_尹金良(1).pdf | 19.36 | `422f2ee116262c5b22a1d2a20fe8dc2548e1e96d0c51d357002e5ee1d3b1e9e2` |

## Cleaning applied in the manuscript

* 550 -> 519: one row kept per duplicate five-gas vector (29 rows removed) and the pair of records with conflicting labels (HTO vs PD) dropped (2 rows).
* 504 -> 387: the 117 records whose five-gas vector also occurs in the main collection are removed.
* Label mapping of the independent collection: 1 = LTO, 2 = HTO, 3 = LED, 4 = HED, 5 = PD, 6 = Normal; verified on the 117 overlapping records (115 agree, 2 differ).
* The 570-record collection carries eight labels (the six classes plus 火花放电 and 电弧放电, 10 records each); only the six classes used in the manuscript are retained.

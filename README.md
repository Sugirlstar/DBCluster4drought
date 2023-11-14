# DBCluster4drought
4D DBCluster algorithm for identifying drought patch

The results of the code should be looks like (four seperate figures):
![image](https://github.com/Sugirlstar/DBCluster4drought/assets/76802881/645c10ac-5994-4a98-be2e-b3f1fe1d582b)

---
Update 2023/11/14
* added the function: set a condition: the point with SPI>-1 cannot be a core point.
* Changes: modified the source code from https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/cluster/_dbscan.py (renamed as _dbscanDR.py)
* To use: change the [__init__.py] file and add the [_dbscanDR.py] within the [cluster] folder (mine is at the path of E:\Lib\site-packages\sklearn\cluster)
  

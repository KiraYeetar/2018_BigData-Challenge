# 2018中国高校大数据挑战赛

复赛A榜 9113，B榜 9122
<br/><br/>
本人新人，代码结构和流程简单易懂，欢迎交流与学习~~

---
**1. 代码浏览手册**

总共7份代码<br/><br/>
编号1 用来存工具函数，是运行之后代码的基础<br/><br/>
编号2 用来得到标签数据集，有几个窗口就存几列<br/><br/>
编号3-6 按照文件名提取特征<br/><br/>
编号7 跑模型,XGBOOST


**2. 划分数据集**

Train11   用户1-20    标签21-27 
Train10   用户1-19    标签20-26
Train9    用户1-18    标签19-25
Train8    用户1-17    标签18-24
Train7    用户1-16    标签17-23
Train6    用户1-15    标签16-22
Train5    用户1-14    标签15-21
Train4    用户1-13    标签14-20
Train3    用户1-12    标签13-19
Train2    用户1-11    标签12-18
Train1    用户1-10    标签11-17

这个数据集划分比较有意思，因为不仅窗口划分密集且从10号开始划分，而且放弃了离测试集“最近的”三个窗口（21-23号），按照平常的套路是有问题的。
但是这个模型的主要思想是“修正”，是为了处理21-23号的异常数据，在模型只学习正常数据的情况下参与模型融合，极为有效的加强融合的差异性。单模型复赛AB榜能达到913和922。

**3. 特征工程**

因为初赛模型的不稳定，外加这个模型数据划分的冒险性，所以在特征上只选择了最为基础的特征，并且没有做衰减。一切以求稳为主。

1）属性特征（设备号、注册类型... ...）
2）基本计数特征（活动次数等，page/action计数... ...）
2）时间特征（注册时间，各表时间的均值、方差、最大... ...）
3）时间差特征（第二点特征与窗口最大最小时间、注册时间的差值）
4）时间间隔特征（时间间隔的均值、最大... ...）


**4. 模型选择**
<br/><br/>XGBoost
```
params = {'booster': 'gbtree',
          'objective':'rank:pairwise',
          'eta': 0.03,
          'max_depth': 5,
          'colsample_bytree': 0.8,
          'subsample': 0.8,
          'min_child_weight': 16}
 num__round=1500
```

# 集成学习


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [集成学习](#集成学习)
  - [基本原理](#基本原理)
    - [Ensemble selection](#ensemble-selection)
    - [Stacking](#stacking)
    - [参考资料](#参考资料)

<!-- /code_chunk_output -->


## 基本原理
在集成学习中，系统通过组合多个模型而不是使用单个模型，来改善机器学习预测效果。 用多个模型的组合代替单个模型，可以增加最终模型的鲁棒性与改善过拟合现象，且多个模型组合后的预测效果通常优于单个模型的预测效果。在集成学习中，各个模型的权重计算主要有以下四种方法，贝叶斯优化（Bayesian optimization），stacking，gradient-free numerical optimization，和ensemble selection。其中**stacking**和**ensemble selection**主要被广泛应用，以下重点介绍stacking与ensemble selection方法。

### Ensemble selection 
![Ensemble selection](../Images/EnsembleSelection.png)
Ensemble selection是一种基于贪心学习的集成学习方法，其具体步骤如下：
1. 初始化一个空的ensemble；
2. 根据设定的评价指标，将在验证集（validation set或hillclimb）上评价指标最优的模型加入到ensemble中；
3. 重复步骤2，直到到达设定的迭代次数，或所有模型均被使用；
4. 返回在验证集上基于目标评价指标上最优的ensemble。
Ensemble selection在加入模型时，模型的权重为1，在添加模型时，允许重复。

### Stacking






### 参考资料
- [What is Bagging vs Boosting in Machine Learning?](https://www.projectpro.io/article/bagging-vs-boosting-in-machine-learning/579#mcetoc_1fvcc9ijdb)
- [集成学习](https://zhuanlan.zhihu.com/p/105038453)
- [Ensemble selection](https://dl.acm.org/doi/abs/10.1145/1015330.1015432)
- [Stacking](https://www.researchgate.net/publication/222467943_Stacked_Generalization)
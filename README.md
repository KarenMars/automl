# autoMLresearch


<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->

<!-- code_chunk_output -->

- [autoMLresearch](#automlresearch)
  - [自动化算法](#自动化算法)
    - [集成模型 (Ensemble selection)](#集成模型-ensemble-selection)
      - [基本原理](#基本原理)
      - [相关实现](#相关实现)
    - [神经架构搜索（Neural architecture search NAS）](#神经架构搜索neural-architecture-search-nas)
      - [基本原理](#基本原理-1)
      - [相关实现](#相关实现-1)
        - [平台](#平台)
        - [工具（开源项目）](#工具开源项目)
      - [论文](#论文)
    - [元学习（Meta-learning）](#元学习meta-learning)
      - [基本原理](#基本原理-2)
      - [相关实现](#相关实现-2)
        - [平台](#平台-1)
        - [工具（开源项目）](#工具开源项目-1)
    - [迁移学习（Transfer learning）](#迁移学习transfer-learning)
      - [基本原理](#基本原理-3)
      - [相关实现](#相关实现-3)
        - [平台](#平台-2)
        - [工具（开源项目）](#工具开源项目-2)
    - [其他自动化算法相关](#其他自动化算法相关)
    - [参考资料](#参考资料)

<!-- /code_chunk_output -->



## 自动化算法
### 集成模型 (Ensemble selection)
#### 基本原理

[集成学习原理](https://git.iflytek.com/hangji/automlresearch/-/blob/master/Theory/ensemble_selection.md)

#### 相关实现

[集成学习相关实现](https://git.iflytek.com/hangji/automlresearch/-/blob/master/Implementations/Ensemble.md)



### 神经架构搜索（Neural architecture search NAS）
#### 基本原理
[神经架构搜索](https://git.iflytek.com/hangji/automlresearch/-/blob/master/Theory/NAS.md)


#### 相关实现
##### 平台
##### 工具（开源项目）
1. [Microsoft NNI](https://github.com/microsoft/nni) 
   - Demo
     - [手写体数字识别任务的神经网络架构搜索](https://git.iflytek.com/hangji/automlresearch/-/blob/master/Demo/NAS/hello_nas.ipynb)
     - [基于DARTS的神经网络架构搜索](https://git.iflytek.com/hangji/automlresearch/-/blob/master/Demo/NAS/darts.ipynb)

#### 论文
- [Primer: Searching for Efficient Transformers for Language Modeling](https://arxiv.org/pdf/2109.08668.pdf)
- [DARTS](https://arxiv.org/abs/1806.09055) (NAS领域经典论文)

### 元学习（Meta-learning）

#### 基本原理


#### 相关实现
##### 平台
- Google Cloud AutoML
##### 工具（开源项目）
- Torchmeta


### 迁移学习（Transfer learning）

#### 基本原理
#### 相关实现
##### 平台
- Google Cloud AutoML
##### 工具（开源项目）

### 其他自动化算法相关
- 自动化优化器选择
- 自动化激活函数
- 自动化Dropout

### 参考资料

- [什么是AutoML](https://www.microsoft.com/zh-cn/videoplayer/embed/RE2Xc9t?postJsllMsg=true&autoCaptions=zh-cn)
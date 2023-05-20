# pybind11-OpenKE

基于 [OpenKE-PyTorch](https://github.com/thunlp/OpenKE/tree/OpenKE-PyTorch) 开发的知识表示学习包，底层数据处理利用 C++ 实现，使用 [pybind11](https://github.com/pybind/pybind11) 实现 C++ 和 Python 的交互。

## New Features

- 利用 C++ 重写底层数据处理，利用 C++11 的线程库实现并行，进而能够做到跨平台 (Windows, Linux).

- 利用 pybind11 实现 Python 和 C++ 的交互.

## OpenKE-PyTorch

>An Open-source Framework for Knowledge Embedding implemented with PyTorch.
>
>This is an Efficient implementation based on PyTorch for knowledge representation learning (KRL). We use C++ to implement some underlying operations such as data preprocessing and negative sampling. For each specific model, it is implemented by PyTorch with Python interfaces so that there is a convenient platform to run models on GPUs.
>
>Models:
>
>- RESCAL: [A Three-Way Model for Collective Learning on Multi-Relational Data](https://icml.cc/Conferences/2011/papers/438_icmlpaper.pdf) .
>
>- TransE: [Translating Embeddings for Modeling Multi-relational Data](https://proceedings.neurips.cc/paper_files/paper/2013/hash/1cecc7a77928ca8133fa24680a88d2f9-Abstract.html) .
>
>- TransH: [Knowledge Graph Embedding by Translating on Hyperplanes](https://ojs.aaai.org/index.php/AAAI/article/view/8870) .
>
>- DistMult: [Embedding Entities and Relations for Learning and Inference in Knowledge Bases](https://arxiv.org/abs/1412.6575) .
>
>- TransR: [Learning Entity and Relation Embeddings for Knowledge Graph Completion](https://ojs.aaai.org/index.php/AAAI/article/view/9491) .
>
>- TransD: [Knowledge Graph Embedding via Dynamic Mapping Matrix](https://aclanthology.org/P15-1067/) .
>
>- HolE: [Holographic Embeddings of Knowledge Graphs](https://ojs.aaai.org/index.php/AAAI/article/view/10314) .
>
>- ComplEx: [Complex Embeddings for Simple Link Prediction](https://arxiv.org/abs/1606.06357) .
>
>- ANALOGY: [Analogical Inference for Multi-relational Embeddings](https://proceedings.mlr.press/v70/liu17d.html) .
>
>- SimplE: [SimplE Embedding for Link Prediction in Knowledge Graphs](https://proceedings.neurips.cc/paper_files/paper/2018/hash/b2ab001909a8a6f04b51920306046ce5-Abstract.html) .
>
>- RotatE: [RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space](https://openreview.net/forum?id=HkgEQnRqYQ) .

## Experimental Settings

>For each test triplet, the head is removed and replaced by each of the entities from the entity set in turn. The scores of those corrupted triplets are first computed by the models and then sorted by the order. Then, we get the rank of the correct entity. This whole procedure is also repeated by removing those tail entities. We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10, Hits@3, Hits@1). The mean rank (MRR) and mean reciprocal rank (MRR) of the test triplets under this setting are also reported.
>
>Because some corrupted triplets may be in the training set and validation set. In this case, those corrupted triplets may be ranked above the test triplet, but this should not be counted as an error because both triplets are true. Hence, we remove those corrupted triplets appearing in the training, validation or test set, which ensures the corrupted triplets are not in the dataset. We report the proportion of those correct entities ranked in the top 10/3/1 (Hits@10 (filter), Hits@3(filter), Hits@1(filter)) under this setting. The mean rank (MRR (filter)) and mean reciprocal rank (MRR (filter)) of the test triplets under this setting are also reported.
>
>More details of the above-mentioned settings can be found from the paper [TransE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf).
>
>For those large-scale entity sets, to corrupt all entities with the whole entity set is time-costing. Hence, we also provide the experimental setting named "[type constraint](https://www.dbs.ifi.lmu.de/~krompass/papers/TypeConstrainedRepresentationLearningInKnowledgeGraphs.pdf)" to corrupt entities with some limited entity sets determining by their relations.

## Experiments

>We have provided the hyper-parameters of some models to achieve the state-of-the-art performace (Hits@10 (filter)) on FB15K237 and WN18RR. These scripts can be founded in the folder "./examples/". Up to now, these models include TransE, TransH, TransR, TransD, DistMult, ComplEx. The results of these models are as follows,

|Model|WN18RR|FB15K237|WN18RR (Paper\*)|FB15K237 (Paper\*)|
|:-:|:-:|:-:|:-:|:-:|
|TransE|0.512|0.476|0.501|0.486|
|TransH|0.507|0.490|-|-|
|TransR|0.519|0.511|-|-|
|TransD|0.508|0.487|-|-|
|DistMult|0.479|0.419|0.49|0.419|
|ComplEx|0.485|0.426|0.51|0.428|
|ConvE|0.506|0.485|0.52|0.501|
|RotatE|0.549|0.479|-|0.480|
|RotatE(+adv)|0.565|0.522|0.571|0.533|

## Installation

1. Install [PyTorch](https://pytorch.org/get-started/locally/).

2. Clone the pybind11-OpenKE-PyTorch branch.

```bash
git clone -b pybind11-OpenKE-PyTorch git@github.com:LuYF-Lemon-love/pybind11-OpenKE.git --depth 1
cd pybind11-OpenKE/
cd pybind11_ke/
```

3. Compile C++ files.

```bash
bash make.sh
```

4. Quick Start.

```bash
cd ../
cp examples/train_transe_FB15K237.py ./
python train_transe_FB15K237.py
```

## Data

>* For training, datasets contain three files:
>
>   train2id.txt: training file, the first line is the number of triples for training. Then the following lines are all in the format ***(e1, e2, rel)*** which indicates there is a relation ***rel*** between ***e1*** and ***e2*** . **Note that train2id.txt contains ids from entitiy2id.txt and relation2id.txt instead of the names of the entities and relations. If you use your own datasets, please check the format of your training file. Files in the wrong format may cause segmentation fault.**
>
>   entity2id.txt: all entities and corresponding ids, one per line. The first line is the number of entities.
>
>   relation2id.txt: all relations and corresponding ids, one per line. The first line is the number of relations.
>
>* For testing, datasets contain additional two files (totally five files):
>
>   test2id.txt: testing file, the first line is the number of triples for testing. Then the following lines are all in the format ***(e1, e2, rel)*** .
>
>   valid2id.txt: validating file, the first line is the number of triples for validating. Then the following lines are all in the format ***(e1, e2, rel)*** .
>
>   type_constrain.txt: type constraining file, the first line is the number of relations. Then the following lines are type constraints for each relation. For example, the relation with id 1200 has 4 types of head entities, which are 3123, 1034, 58 and 5733. The relation with id 1200 has 4 types of tail entities, which are 12123, 4388, 11087 and 11088. You can get this file through **n-n.py** in folder benchmarks/FB15K.

## Files

- [benchmarks](./benchmarks/): 数据集.

- [examples](./examples/): 例子.

- [openke](./openke/): 知识表示学习包.

## Reference

[1] Xu Han, Shulin Cao, Xin Lv, Yankai Lin, Zhiyuan Liu, Maosong Sun, and Juanzi Li. 2018. OpenKE: An Open Toolkit for Knowledge Embedding. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 139–144, Brussels, Belgium. Association for Computational Linguistics.

[2] [pybind11 — Seamless operability between C++11 and Python](https://github.com/pybind/pybind11).

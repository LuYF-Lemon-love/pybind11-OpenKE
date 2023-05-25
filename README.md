# pybind11-OpenKE

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

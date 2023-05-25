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
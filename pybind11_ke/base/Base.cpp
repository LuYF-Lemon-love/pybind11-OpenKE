// pybind11-ke/base/Base.cpp
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 23, 2023
// 
// 该头文件定义了 Python 和 C++ 的交互接口.

#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include <cstdlib>
#include <pthread.h>
#include <thread>

// defined in Setting.h
extern "C"
void setValidPath(char *path);

// defined in Setting.h
extern "C"
void setTestPath(char *path);

// defined in Setting.h
extern "C"
void setOutPath(char *path);

// defined in Setting.h
extern "C"
INT getWorkThreads();

// defined in Setting.h
extern "C"
INT getTripleTotal();

// defined in Setting.h
extern "C"
INT getValidTotal();

// Python 与 C++ 之间传递的数据结构
// id: 线程 ID
// batch_h_py: head entity
// batch_t_py: tail entity
// batch_r_py: relation
// batch_y_py: label
// batchSize: batch size
// negRate: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail).
// negRelRate: 对于每一个正三元组, 构建的负三元组的个数, 替换 relation.
// mode: 控制构建的方式, mode = 0 and bernFlag = True, 起用 TransH 方式构建负三元组.
// filter_flag: 提出于 TransE, 用于更好的构建负三元组, used in corrupt_head, corrupt_tail, corrupt_rel.
// filter_flag: 源代码中好像没有用到.
// p: 用于构建负三元组 (used in corrupt_rel)
// val_loss: val_loss == false (构建负三元组), else 不构建负三元组

// 获得 1 batch 训练数据
void getBatch(
	INT id,
	py::array_t<INT> batch_h_py, 
	py::array_t<INT> batch_t_py, 
	py::array_t<INT> batch_r_py, 
	py::array_t<REAL> batch_y_py, 
	INT batchSize, 
	INT negRate, 
	INT negRelRate, 
	INT mode,
	bool filter_flag,
	bool p, 
	bool val_loss
) {
	auto batch_h = batch_h_py.mutable_unchecked<1>();
	auto batch_t = batch_t_py.mutable_unchecked<1>();
	auto batch_r = batch_r_py.mutable_unchecked<1>();
	auto batch_y = batch_y_py.mutable_unchecked<1>();
	// 线程 id 负责生成 [lef, rig) 范围的训练数据
	INT lef, rig;
	if (batchSize % workThreads == 0) {
		lef = id * (batchSize / workThreads);
		rig = (id + 1) * (batchSize / workThreads);
	} else {
		lef = id * (batchSize / workThreads + 1);
		rig = (id + 1) * (batchSize / workThreads + 1);
		if (rig > batchSize) rig = batchSize;
	}
	REAL prob = 500;
	if (val_loss == false) {
		for (INT batch = lef; batch < rig; batch++) {
			// 正三元组
			INT i = rand_max(id, trainTotal);
			batch_h(batch) = trainList[i].h;
			batch_t(batch) = trainList[i].t;
			batch_r(batch) = trainList[i].r;
			batch_y(batch) = 1;
			// batch + batchSize: 第一个负三元组生成的位置
			INT last = batchSize;
			// 负采样 entity
			for (INT times = 0; times < negRate; times ++) {
				if (mode == 0){
					// TransH 负采样策略
					if (bernFlag)
						prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
					if (randd(id) % 1000 < prob) {
						batch_h(batch + last) = trainList[i].h;
						batch_t(batch + last) = corrupt_head(id, trainList[i].h, trainList[i].r);
						batch_r(batch + last) = trainList[i].r;
					} else {
						batch_h(batch + last) = corrupt_tail(id, trainList[i].t, trainList[i].r);
						batch_t(batch + last) = trainList[i].t;
						batch_r(batch + last) = trainList[i].r;
					}
					batch_y(batch + last) = -1;
					// 下一负三元组的位置
					last += batchSize;
				} else {
					if(mode == -1){
						batch_h(batch + last) = corrupt_tail(id, trainList[i].t, trainList[i].r);
						batch_t(batch + last) = trainList[i].t;
						batch_r(batch + last) = trainList[i].r;
					} else {
						batch_h(batch + last) = trainList[i].h;
						batch_t(batch + last) = corrupt_head(id, trainList[i].h, trainList[i].r);
						batch_r(batch + last) = trainList[i].r;
					}
					batch_y(batch + last) = -1;
					last += batchSize;
				}
			}
			// 负采样 relation
			for (INT times = 0; times < negRelRate; times++) {
				batch_h(batch + last) = trainList[i].h;
				batch_t(batch + last) = trainList[i].t;
				batch_r(batch + last) = corrupt_rel(id, trainList[i].h, trainList[i].t, trainList[i].r, p);
				batch_y(batch + last) = -1;
				last += batchSize;
			}
		}
	}
	else
	{
		for (INT batch = lef; batch < rig; batch++)
		{
			batch_h(batch) = validList[batch].h;
			batch_t(batch) = validList[batch].t;
			batch_r(batch) = validList[batch].r;
			batch_y(batch) = 1;
		}
	}
}

// 真正的数据处理函数
void sampling(
		py::array_t<INT> batch_h, 
		py::array_t<INT> batch_t, 
		py::array_t<INT> batch_r, 
		py::array_t<REAL> batch_y, 
		INT batchSize, 
		INT negRate = 1, 
		INT negRelRate = 0, 
		INT mode = 0,
		bool filter_flag = true,
		bool p = false, 
		bool val_loss = false
) {
	std::vector<std::thread> threads;
    for (INT id = 0; id < workThreads; id++)
    {
        threads.emplace_back(getBatch, id, batch_h,
			batch_t, batch_r, batch_y, batchSize,
			negRate, negRelRate, mode, filter_flag,
			p, val_loss);
    }
    for(auto& entry: threads)
        entry.join();
}

PYBIND11_MODULE(base, m) {
	m.doc() = "The underlying data processing module of pybind11-OpenKE is powered by pybind11.";

	m.def("sampling", &sampling, "sample function",
		py::arg("batch_h").noconvert(), py::arg("batch_t").noconvert(),
		py::arg("batch_r").noconvert(), py::arg("batch_y").noconvert(),
		py::arg("batchSize"), py::arg("bnegRate") = 1,
		py::arg("negRelRate") = 0, py::arg("mode") = 0,
		py::arg("filter_flag") = true, py::arg("p") = false,
		py::arg("val_loss") = false,
        py::call_guard<py::gil_scoped_release>());

	m.def("setInPath", &setInPath);
	m.def("setTrainPath", &setTrainPath);
	m.def("setEntPath", &setEntPath);
	m.def("setRelPath", &setRelPath);
	m.def("setBern", &setBern);
	m.def("setWorkThreads", &setWorkThreads);
	m.def("randReset", &randReset);
	m.def("importTrainFiles", &importTrainFiles);
	m.def("getRelationTotal", &getRelationTotal);
	m.def("getEntityTotal", &getEntityTotal);
	m.def("getTrainTotal", &getTrainTotal);

	m.def("importTestFiles", &importTestFiles);
	m.def("importTypeFiles", &importTypeFiles);
	m.def("getTestTotal", &getTestTotal);
	m.def("getHeadBatch", &getHeadBatch, "对于测试集中的给定三元组, 用所有实体替换 head, 返回所有三元组.",
		py::arg("ph_py").noconvert(), py::arg("pt_py").noconvert(),
		py::arg("pr_py").noconvert());
	m.def("getTailBatch", &getTailBatch, "对于测试集中的给定三元组, 用所有实体替换 tail, 返回所有三元组.",
		py::arg("ph_py").noconvert(), py::arg("pt_py").noconvert(),
		py::arg("pr_py").noconvert());
	m.def("initTest", &initTest);

	m.def("testHead", &testHead, "替换 head, 评估 head 的 rank.",
		py::arg("con_py").noconvert(), py::arg("lastHead"),
		py::arg("type_constrain") = false);
	m.def("testTail", &testTail, "替换 tail, 评估 tail 的 rank.",
		py::arg("con_py").noconvert(), py::arg("lastTail"),
		py::arg("type_constrain") = false);
	m.def("test_link_prediction", &test_link_prediction, "链接预测入口函数",
		py::arg("type_constrain") = false);
	m.def("getTestLinkMRR", &getTestLinkMRR, "return MRR",
		py::arg("type_constrain") = false);
	m.def("getTestLinkMR", &getTestLinkMR, "return MR",
		py::arg("type_constrain") = false);
	m.def("getTestLinkHit10", &getTestLinkHit10, "return Hit10",
		py::arg("type_constrain") = false);
	m.def("getTestLinkHit3", &getTestLinkHit3, "return Hit3",
		py::arg("type_constrain") = false);
	m.def("getTestLinkHit1", &getTestLinkHit1, "return Hit1",
		py::arg("type_constrain") = false);
}
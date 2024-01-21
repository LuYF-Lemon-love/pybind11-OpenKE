// pybind11-ke/base/Base.cpp
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Dec 30, 2023
// 
// 该头文件定义了 Python 和 C++ 的交互接口.

#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include <thread>
#include <stdexcept>

// Python 与 C++ 之间传递的数据结构
// id: 线程 ID
// batch_h_py: head entity
// batch_t_py: tail entity
// batch_r_py: relation
// batch_y_py: label
// batch_size: batch size
// neg_ent: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail).
// neg_rel: 对于每一个正三元组, 构建的负三元组的个数, 替换 relation.
// mode: 控制构建的方式, mode = 0 and bern = True, 起用 TransH 方式构建负三元组.
//		mode = 1 : 只替换头实体; mode = -1 : 只替换尾实体.

// 获得 1 batch 训练数据
void get_bacth(
	INT id,
	py::array_t<INT> batch_h_py, 
	py::array_t<INT> batch_t_py, 
	py::array_t<INT> batch_r_py, 
	py::array_t<REAL> batch_y_py, 
	INT batch_size, 
	INT neg_ent, 
	INT neg_rel, 
	INT mode
) {
	auto batch_h = batch_h_py.mutable_unchecked<1>();
	auto batch_t = batch_t_py.mutable_unchecked<1>();
	auto batch_r = batch_r_py.mutable_unchecked<1>();
	auto batch_y = batch_y_py.mutable_unchecked<1>();
	// 线程 id 负责生成 [lef, rig) 范围的训练数据
	INT lef, rig;
	if (batch_size % work_threads == 0) {
		lef = id * (batch_size / work_threads);
		rig = (id + 1) * (batch_size / work_threads);
	} else {
		lef = id * (batch_size / work_threads + 1);
		rig = (id + 1) * (batch_size / work_threads + 1);
		if (rig > batch_size) rig = batch_size;
	}
	REAL prob = 500;
	for (INT batch = lef; batch < rig; batch++) {
		// 正三元组
		INT i = rand_max(id, train_total);
		try {
			batch_h(batch) = train_list.at(i).h;
			batch_t(batch) = train_list.at(i).t;
			batch_r(batch) = train_list.at(i).r;
			batch_y(batch) = 1;
		} catch (std::out_of_range const& exc) {
			std::cout << exc.what() << std::endl;
			std::cout << i << std::endl;
			std::cout << "rand_max" << std::endl;
		}
		// std:: cout << i << " ";
		// batch + batch_size: 第一个负三元组生成的位置
		INT last = batch_size;
		// 负采样 entity
		for (INT times = 0; times < neg_ent; times ++) {
			if (mode == 0){
				// TransH 负采样策略
				if (bern)
					prob = 1000 * hpt.at(train_list.at(i).r) / (hpt.at(train_list.at(i).r) + tph.at(train_list.at(i).r));
				if (rand_max(id, 1000) < prob) {
					batch_h(batch + last) = train_list.at(i).h;
					try {
						batch_t(batch + last) = corrupt_with_head(id, train_list.at(i).h, train_list.at(i).r);
					} catch (std::out_of_range const& exc) {
						std::cout << exc.what() << std::endl;
						std::cout << "corrupt_with_head" << std::endl;
					}
					batch_r(batch + last) = train_list.at(i).r;
				} else {
					try {
						batch_h(batch + last) = corrupt_with_tail(id, train_list.at(i).t, train_list.at(i).r);
					} catch (std::out_of_range const& exc) {
						std::cout << exc.what() << std::endl;
						std::cout << "corrupt_with_tail" << std::endl;
					}
					batch_t(batch + last) = train_list.at(i).t;
					batch_r(batch + last) = train_list.at(i).r;
				}
				batch_y(batch + last) = -1;
				// 下一负三元组的位置
				last += batch_size;
			} else {
				if(mode == 1){
					try {
						batch_h(batch + last) = corrupt_with_tail(id, train_list.at(i).t, train_list.at(i).r);
					} catch (std::out_of_range const& exc) {
						std::cout << exc.what() << std::endl;
						std::cout << "corrupt_with_tail" << std::endl;
					}
					batch_t(batch + last) = train_list.at(i).t;
					batch_r(batch + last) = train_list.at(i).r;
				} else if (mode == -1){
					batch_h(batch + last) = train_list.at(i).h;
					try {
						batch_t(batch + last) = corrupt_with_head(id, train_list.at(i).h, train_list.at(i).r);
					} catch (std::out_of_range const& exc) {
						std::cout << exc.what() << std::endl;
						std::cout << "corrupt_with_head" << std::endl;
					}
					batch_r(batch + last) = train_list.at(i).r;
				}
				batch_y(batch + last) = -1;
				last += batch_size;
			}
		}
		// 负采样 relation
		for (INT times = 0; times < neg_rel; times++) {
			batch_h(batch + last) = train_list.at(i).h;
			batch_t(batch + last) = train_list.at(i).t;
			batch_r(batch + last) = corrupt_rel(id, train_list.at(i).h, train_list.at(i).t);
			batch_y(batch + last) = -1;
			last += batch_size;
		}
	}
}

// 真正的数据处理函数
void sampling(
		py::array_t<INT> batch_h, 
		py::array_t<INT> batch_t, 
		py::array_t<INT> batch_r, 
		py::array_t<REAL> batch_y, 
		INT batch_size, 
		INT neg_ent = 1, 
		INT neg_rel = 0, 
		INT mode = 0
) {
	std::vector<std::thread> threads;
    for (INT id = 0; id < work_threads; id++)
    {
        threads.emplace_back(get_bacth, id, batch_h,
			batch_t, batch_r, batch_y, batch_size,
			neg_ent, neg_rel, mode);
    }
    for(auto& entry: threads)
        entry.join();
}

PYBIND11_MODULE(base, m) {
	m.doc() = "The underlying data processing module of pybind11-OpenKE is powered by pybind11.";

	m.def("sampling", &sampling, "sample function",
		py::arg("batch_h").noconvert(), py::arg("batch_t").noconvert(),
		py::arg("batch_r").noconvert(), py::arg("batch_y").noconvert(),
		py::arg("batch_size"), py::arg("neg_ent") = 1,
		py::arg("neg_rel") = 0, py::arg("mode") = 0,
        py::call_guard<py::gil_scoped_release>());

	m.def("set_in_path", &set_in_path);
	m.def("set_ent_path", &set_ent_path);
	m.def("set_rel_path", &set_rel_path);
	m.def("set_train_path", &set_train_path);
	m.def("set_valid_path", &set_valid_path);
	m.def("set_test_path", &set_test_path);

	m.def("set_bern", &set_bern);
	m.def("set_work_threads", &set_work_threads);
	m.def("rand_reset", &rand_reset);

	m.def("read_train_files", &read_train_files);
	m.def("get_relation_total", &get_relation_total);
	m.def("get_entity_total", &get_entity_total);
	m.def("get_train_total", &get_train_total);

	m.def("read_test_files", &read_test_files);
	m.def("read_type_files", &read_type_files);
	m.def("get_test_total", &get_test_total);
	m.def("get_valid_total", &get_valid_total);
	
	m.def("get_head_batch", &get_head_batch, "对于测试集中的给定三元组, 用所有实体替换 head, 返回所有三元组.",
		py::arg("ph_py").noconvert(), py::arg("pt_py").noconvert(),
		py::arg("pr_py").noconvert(), py::arg("sampling_mode").noconvert());
	m.def("get_tail_batch", &get_tail_batch, "对于测试集中的给定三元组, 用所有实体替换 tail, 返回所有三元组.",
		py::arg("ph_py").noconvert(), py::arg("pt_py").noconvert(),
		py::arg("pr_py").noconvert(), py::arg("sampling_mode").noconvert());
	
	m.def("init_test", &init_test);

	m.def("test_head", &test_head, "替换 head, 评估 head 的 rank.",
		py::arg("con_py").noconvert(), py::arg("type_constrain") = false,
		py::arg("sampling_mode") = "link_test");
	m.def("test_tail", &test_tail, "替换 tail, 评估 tail 的 rank.",
		py::arg("con_py").noconvert(), py::arg("type_constrain") = false,
		py::arg("sampling_mode") = "link_test");

	m.def("test_link_prediction", &test_link_prediction, "链接预测入口函数",
		py::arg("type_constrain") = false, py::arg("sampling_mode").noconvert());
	
	m.def("get_test_link_MR", &get_test_link_MR, "return MR",
		py::arg("type_constrain") = false);
	m.def("get_test_link_MRR", &get_test_link_MRR, "return MRR",
		py::arg("type_constrain") = false);
	m.def("get_test_link_Hit1", &get_test_link_Hit1, "return Hit1",
		py::arg("type_constrain") = false);
	m.def("get_test_link_Hit3", &get_test_link_Hit3, "return Hit3",
		py::arg("type_constrain") = false);
	m.def("get_test_link_Hit10", &get_test_link_Hit10, "return Hit10",
		py::arg("type_constrain") = false);
}
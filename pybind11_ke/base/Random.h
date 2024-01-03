// pybind11-ke/base/Random.h
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Dec 30, 2023
// 
// 该头文件定义了一些随机函数.

#ifndef RANDOM_H
#define RANDOM_H
#include <random>

// 所有线程的随机种子.
std::random_device rd;
std::vector<std::mt19937_64> gens;
std::vector<std::uniform_int_distribution<INT>> dists;

std::mt19937_64 gen{rd()};
std::uniform_int_distribution<INT> dist{0, 10};

// 生成一个 [a,b) 范围内的随机整数.
INT rand(INT a, INT b){
	decltype(dist)::param_type param{a, b-1};
	return dist(gen, param);
}

// 重新设定所有线程的随机种子.
void rand_reset() {
	for (INT i = 0; i < work_threads; i++) {
		gens.emplace_back(rd());
		dists.emplace_back(0, 10);
	}
}

// 为 id 对应的线程生成一个 [0,x) 范围内的随机整数.
INT rand_max(INT id, INT x) {
	std::uniform_int_distribution<INT>::param_type param{0, x-1};
	return dists[id](gens[id], param);
}

#endif

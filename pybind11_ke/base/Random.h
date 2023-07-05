// pybind11-ke/base/Random.h
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 24, 2023
// 
// 该头文件定义了一些随机函数.

#ifndef RANDOM_H
#define RANDOM_H
#include "Setting.h"

// 所有线程的随机种子.
unsigned long long *next_random;

// 重新设定所有线程的随机种子.
void rand_reset() {
	next_random = (unsigned long long *)calloc(work_threads, sizeof(unsigned long long));
	for (INT i = 0; i < work_threads; i++)
		next_random[i] = rand();
}

// 为 id 对应的线程生成下一个随机整数.
unsigned long long randd(INT id) {
	next_random[id] = next_random[id] * (unsigned long long)(25214903917) + 11;
	return next_random[id];
}

// 为 id 对应的线程生成一个 [0,x) 范围内的随机整数.
INT rand_max(INT id, INT x) {
	INT res = randd(id) % x;
	while (res < 0)
		res += x;
	return res;
}

// 为 id 对应的线程生成一个 [a,b) 范围内的随机整数.
INT rand(INT a, INT b){
	return (rand() % (b-a)) + a;
}

#endif

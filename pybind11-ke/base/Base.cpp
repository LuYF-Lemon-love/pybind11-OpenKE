// pybind11-ke/base/Base.cpp
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 10, 2023
// 
// 该头文件定义了 Python 和 C++ 的交互接口.

#include "Setting.h"
#include "Random.h"
#include "Reader.h"
#include "Corrupt.h"
#include "Test.h"
#include <cstdlib>
#include <pthread.h>

// defined in Setting.h
extern "C"
void setInPath(char *path);

// defined in Setting.h
extern "C"
void setTrainPath(char *path);

// defined in Setting.h
extern "C"
void setValidPath(char *path);

// defined in Setting.h
extern "C"
void setTestPath(char *path);

// defined in Setting.h
extern "C"
void setEntPath(char *path);

// defined in Setting.h
extern "C"
void setRelPath(char *path);

// defined in Setting.h
extern "C"
void setOutPath(char *path);

// defined in Setting.h
extern "C"
void setWorkThreads(INT threads);

// defined in Setting.h
extern "C"
void setBern(INT con);

// defined in Setting.h
extern "C"
INT getWorkThreads();

// defined in Setting.h
extern "C"
INT getEntityTotal();

// defined in Setting.h
extern "C"
INT getRelationTotal();

// defined in Setting.h
extern "C"
INT getTripleTotal();

// defined in Setting.h
extern "C"
INT getTrainTotal();

// defined in Setting.h
extern "C"
INT getTestTotal();

// defined in Setting.h
extern "C"
INT getValidTotal();

// defined in Random.h
extern "C"
void randReset();

// defined in Reader.h
extern "C"
void importTrainFiles();

// Python 与 C++ 之间传递的数据结构
// id: 线程 ID
// batch_h: head entity
// batch_t: tail entity
// batch_r: relation
// batch_y: label
// batchSize: batch size
// negRate: 对于每一个正三元组, 构建的负三元组的个数, 替换 entity (head + tail).
// negRelRate: 对于每一个正三元组, 构建的负三元组的个数, 替换 relation.
// p: 用于构建负三元组 (used in corrupt_rel)
// val_loss: val_loss == false (构建负三元组), else 不构建负三元组
// mode: 控制构建的方式, mode = 0 and bernFlag = True, 起用 TransH 方式构建负三元组.
// filter_flag: 提出于 TransE, 用于更好的构建负三元组, used in corrupt_head, corrupt_tail, corrupt_rel.
// filter_flag: 源代码中好像没有用到.
struct Parameter {
	INT id;
	INT *batch_h;
	INT *batch_t;
	INT *batch_r;
	REAL *batch_y;
	INT batchSize;
	INT negRate;
	INT negRelRate;
	bool p;
	bool val_loss;
	INT mode;
	bool filter_flag;
};

void* getBatch(void* con) {
	Parameter *para = (Parameter *)(con);
	INT id = para -> id;
	INT *batch_h = para -> batch_h;
	INT *batch_t = para -> batch_t;
	INT *batch_r = para -> batch_r;
	REAL *batch_y = para -> batch_y;
	INT batchSize = para -> batchSize;
	INT negRate = para -> negRate;
	INT negRelRate = para -> negRelRate;
	bool p = para -> p;
	bool val_loss = para -> val_loss;
	INT mode = para -> mode;
	bool filter_flag = para -> filter_flag;
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
			INT i = rand_max(id, trainTotal);
			batch_h[batch] = trainList[i].h;
			batch_t[batch] = trainList[i].t;
			batch_r[batch] = trainList[i].r;
			batch_y[batch] = 1;
			INT last = batchSize;
			for (INT times = 0; times < negRate; times ++) {
				if (mode == 0){
					if (bernFlag)
						prob = 1000 * right_mean[trainList[i].r] / (right_mean[trainList[i].r] + left_mean[trainList[i].r]);
					if (randd(id) % 1000 < prob) {
						batch_h[batch + last] = trainList[i].h;
						batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
						batch_r[batch + last] = trainList[i].r;
					} else {
						batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);
						batch_t[batch + last] = trainList[i].t;
						batch_r[batch + last] = trainList[i].r;
					}
					batch_y[batch + last] = -1;
					last += batchSize;
				} else {
					if(mode == -1){
						batch_h[batch + last] = corrupt_tail(id, trainList[i].t, trainList[i].r);
						batch_t[batch + last] = trainList[i].t;
						batch_r[batch + last] = trainList[i].r;
					} else {
						batch_h[batch + last] = trainList[i].h;
						batch_t[batch + last] = corrupt_head(id, trainList[i].h, trainList[i].r);
						batch_r[batch + last] = trainList[i].r;
					}
					batch_y[batch + last] = -1;
					last += batchSize;
				}
			}
			for (INT times = 0; times < negRelRate; times++) {
				batch_h[batch + last] = trainList[i].h;
				batch_t[batch + last] = trainList[i].t;
				batch_r[batch + last] = corrupt_rel(id, trainList[i].h, trainList[i].t, trainList[i].r, p);
				batch_y[batch + last] = -1;
				last += batchSize;
			}
		}
	}
	else
	{
		for (INT batch = lef; batch < rig; batch++)
		{
			batch_h[batch] = validList[batch].h;
			batch_t[batch] = validList[batch].t;
			batch_r[batch] = validList[batch].r;
			batch_y[batch] = 1;
		}
	}
	pthread_exit(NULL);
}

extern "C"
void sampling(
		INT *batch_h, 
		INT *batch_t, 
		INT *batch_r, 
		REAL *batch_y, 
		INT batchSize, 
		INT negRate = 1, 
		INT negRelRate = 0, 
		INT mode = 0,
		bool filter_flag = true,
		bool p = false, 
		bool val_loss = false
) {
	pthread_t *pt = (pthread_t *)malloc(workThreads * sizeof(pthread_t));
	Parameter *para = (Parameter *)malloc(workThreads * sizeof(Parameter));
	for (INT threads = 0; threads < workThreads; threads++) {
		para[threads].id = threads;
		para[threads].batch_h = batch_h;
		para[threads].batch_t = batch_t;
		para[threads].batch_r = batch_r;
		para[threads].batch_y = batch_y;
		para[threads].batchSize = batchSize;
		para[threads].negRate = negRate;
		para[threads].negRelRate = negRelRate;
		para[threads].p = p;
		para[threads].val_loss = val_loss;
		para[threads].mode = mode;
		para[threads].filter_flag = filter_flag;
		pthread_create(&pt[threads], NULL, getBatch, (void*)(para+threads));
	}
	for (INT threads = 0; threads < workThreads; threads++)
		pthread_join(pt[threads], NULL);

	free(pt);
	free(para);
}

int main() {
	importTrainFiles();
	return 0;
}
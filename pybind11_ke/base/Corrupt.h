// pybind11-ke/base/Corrupt.h
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Dec 30, 2023
// 
// 该头文件定义了破坏三元组的方法.

#ifndef CORRUPT_H
#define CORRUPT_H
#include "Random.h"
#include "Reader.h"

// 用 head 和 relation 构建负三元组，即替换 tail
// 该函数返回负三元组的 tail
INT corrupt_with_head(INT id, INT h, INT r) {
	INT lef, rig, mid, ll, rr;

	// lef: head(h) 在 train_head 中第一次出现的前一个位置
	// rig: head(h) 在 train_head 中最后一次出现的位置
	lef = begin_head[h] - 1;
	rig = end_head[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 >= -> rig，所以 rig 最终在第一个 r 的位置
		if (train_head[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;

	lef = begin_head[h];
	rig = end_head[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 <= -> lef，所以 lef 最终在最后一个 r 的位置
		if (train_head[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;

	// 只能产生 (entity_total - (rr - ll + 1)) 种实体，即去掉训练集中已有的三元组
	INT tmp = rand_max(id, entity_total - (rr - ll + 1));

	// 第一种：tmp 小于第一个 r 对应的 tail
	if (tmp < train_head[ll].t) return tmp;

	// 第二种：tmp 大于最后一个 r 对应的 tail
	if (tmp > train_head[rr].t - rr + ll - 1) return tmp + rr - ll + 1;

	// 第三种：由于 (>= -> rig), (lef + 1 < rig), (tmp + lef - ll + 1)
	// 因此最终返回取值为 (train_head[lef].t, train_head[rig].t) 的 tail
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 类似 tmp > train_head[rr].t - rr + ll - 1
		if (train_head[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	// 类似 tmp + rr - ll + 1
	return tmp + lef - ll + 1;
}

// 用 tail 和 relation 构建负三元组，即替换 head
// 该函数返回负三元组的 head
INT corrupt_with_tail(INT id, INT t, INT r) {
	INT lef, rig, mid, ll, rr;

	// lef: tail(t) 在 train_tail 中第一次出现的前一个位置
	// rig: tail(t) 在 train_tail 中最后一次出现的位置
	lef = begin_tail.at(t) - 1;
	rig = end_tail.at(t);
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 >= -> rig，所以 rig 最终在第一个 r 的位置
		if (train_tail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;

	lef = begin_tail.at(t);
	rig = end_tail.at(t) + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 <= -> lef，所以 lef 最终在最后一个 r 的位置
		if (train_tail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;

	// 只能产生 (entity_total - (rr - ll + 1)) 种实体，即去掉训练集中已有的三元组
	INT tmp = rand_max(id, entity_total - (rr - ll + 1));

	// 第一种：tmp 小于第一个 r 对应的 head
	if (tmp < train_tail[ll].h) return tmp;

	// 第二种：tmp 大于最后一个 r 对应的 head
	if (tmp > train_tail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;

	// 第三种：由于 (>= -> rig), (lef + 1 < rig), (tmp + lef - ll + 1)
	// 因此最终返回取值为 (train_tail[lef].h, train_tail[rig].h) 的 head
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 类似 tmp > train_tail[rr].h - rr + ll - 1
		if (train_tail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	// 类似 tmp + rr - ll + 1
	return tmp + lef - ll + 1;
}

// 用 head 和 tail 构建负三元组，即替换 relation
// 该函数返回负三元组的 relation
INT corrupt_rel(INT id, INT h, INT t) {
	INT lef, rig, mid, ll, rr;

	// lef: head(h) 在 train_rel 中第一次出现的前一个位置
	// rig: head(h) 在 train_rel 中最后一次出现的位置
	lef = begin_rel.at(h) - 1;
	rig = end_rel.at(h);
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 >= -> rig，所以 rig 最终在第一个 t 的位置
		if (train_rel[mid].t >= t) rig = mid; else
		lef = mid;
	}
	ll = rig;

	lef = begin_rel.at(h);
	rig = end_rel.at(h) + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 <= -> lef，所以 lef 最终在最后一个 t 的位置
		if (train_rel[mid].t <= t) lef = mid; else
		rig = mid;
	}
	rr = lef;
	
	// 只能产生 (elationTotal - (rr - ll + 1)) 种关系，即去掉训练集中已有的三元组
	INT	tmp = rand_max(id, relation_total - (rr - ll + 1));

	// 第一种：tmp 小于第一个 t 对应的 relation
	if (tmp < train_rel[ll].r) return tmp;

	// 第二种：tmp 大于最后一个 t 对应的 relation
	if (tmp > train_rel[rr].r - rr + ll - 1) return tmp + rr - ll + 1;
	
	// 第三种：由于 (>= -> rig), (lef + 1 < rig), (tmp + lef - ll + 1)
	// 因此最终返回取值为 (train_rel[lef].t, train_rel[rig].t) 的 tail
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 类似 tmp > train_rel[rr].r - rr + ll - 1
		if (train_rel[mid].r - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	// 类似 tmp + rr - ll + 1
	return tmp + lef - ll + 1;
}

// 检查数据集中是否存在 (h, t, r)
bool _find(INT h, INT t, INT r) {
    INT lef = 0;
    INT rig = triple_total - 1;
    INT mid;
    while (lef + 1 < rig) {
        mid = (lef + rig) >> 1;
        if ((triple_list.at(mid). h < h) || (triple_list.at(mid). h == h && triple_list.at(mid). r < r) || (triple_list.at(mid). h == h && triple_list.at(mid). r == r && triple_list.at(mid). t < t)) lef = mid; else rig = mid;
    }
    if (triple_list.at(lef).h == h && triple_list.at(lef).r == r && triple_list.at(lef).t == t) return true;
    if (triple_list.at(rig).h == h && triple_list.at(rig).r == r && triple_list.at(rig).t == t) return true;
    return false;
}
#endif

// pybind11-ke/base/Corrupt.h
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 9, 2023
// 
// 该头文件定义了破坏三元组的方法.

#ifndef CORRUPT_H
#define CORRUPT_H
#include "Random.h"
#include "Triple.h"
#include "Reader.h"

// 用 head 和 relation 构建负三元组，即替换 tail
// 该函数返回负三元组的 tail
INT corrupt_head(INT id, INT h, INT r) {
	INT lef, rig, mid, ll, rr;

	// lef: head(h) 在 trainHead 中第一次出现的前一个位置
	// rig: head(h) 在 trainHead 中最后一次出现的位置
	lef = lefHead[h] - 1;
	rig = rigHead[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 >= -> rig，所以 rig 最终在第一个 r 的位置
		if (trainHead[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;

	lef = lefHead[h];
	rig = rigHead[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 <= -> lef，所以 lef 最终在最后一个 r 的位置
		if (trainHead[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;

	// 只能产生 (entity_total - (rr - ll + 1)) 种实体，即去掉训练集中已有的三元组
	INT tmp = rand_max(id, entity_total - (rr - ll + 1));

	// 第一种：tmp 小于第一个 r 对应的 tail
	if (tmp < trainHead[ll].t) return tmp;

	// 第二种：tmp 大于最后一个 r 对应的 tail
	if (tmp > trainHead[rr].t - rr + ll - 1) return tmp + rr - ll + 1;

	// 第三种：由于 (>= -> rig), (lef + 1 < rig), (tmp + lef - ll + 1)
	// 因此最终返回取值为 (train_head[lef].t, train_head[rig].t) 的 tail
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 类似 tmp > trainHead[rr].t - rr + ll - 1
		if (trainHead[mid].t - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	// 类似 tmp + rr - ll + 1
	return tmp + lef - ll + 1;
}

// 用 tail 和 relation 构建负三元组，即替换 head
// 该函数返回负三元组的 head
INT corrupt_tail(INT id, INT t, INT r) {
	INT lef, rig, mid, ll, rr;

	// lef: tail(t) 在 trainTail 中第一次出现的前一个位置
	// rig: tail(t) 在 trainTail 中最后一次出现的位置
	lef = lefTail[t] - 1;
	rig = rigTail[t];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 >= -> rig，所以 rig 最终在第一个 r 的位置
		if (trainTail[mid].r >= r) rig = mid; else
		lef = mid;
	}
	ll = rig;

	lef = lefTail[t];
	rig = rigTail[t] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 <= -> lef，所以 lef 最终在最后一个 r 的位置
		if (trainTail[mid].r <= r) lef = mid; else
		rig = mid;
	}
	rr = lef;

	// 只能产生 (entity_total - (rr - ll + 1)) 种实体，即去掉训练集中已有的三元组
	INT tmp = rand_max(id, entity_total - (rr - ll + 1));

	// 第一种：tmp 小于第一个 r 对应的 head
	if (tmp < trainTail[ll].h) return tmp;

	// 第二种：tmp 大于最后一个 r 对应的 head
	if (tmp > trainTail[rr].h - rr + ll - 1) return tmp + rr - ll + 1;

	// 第三种：由于 (>= -> rig), (lef + 1 < rig), (tmp + lef - ll + 1)
	// 因此最终返回取值为 (train_tail[lef].h, train_tail[rig].h) 的 head
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 类似 tmp > trainTail[rr].h - rr + ll - 1
		if (trainTail[mid].h - mid + ll - 1 < tmp)
			lef = mid;
		else 
			rig = mid;
	}
	// 类似 tmp + rr - ll + 1
	return tmp + lef - ll + 1;
}

// 用 head 和 tail 构建负三元组，即替换 relation
// 该函数返回负三元组的 relation
INT corrupt_rel(INT id, INT h, INT t, INT r, bool p = false) {
	INT lef, rig, mid, ll, rr;

	// lef: head(h) 在 trainRel 中第一次出现的前一个位置
	// rig: head(h) 在 trainRel 中最后一次出现的位置
	lef = lefRel[h] - 1;
	rig = rigRel[h];
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 >= -> rig，所以 rig 最终在第一个 t 的位置
		if (trainRel[mid].t >= t) rig = mid; else
		lef = mid;
	}
	ll = rig;

	lef = lefRel[h];
	rig = rigRel[h] + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 二分查找算法变体
		// 由于 <= -> lef，所以 lef 最终在最后一个 t 的位置
		if (trainRel[mid].t <= t) lef = mid; else
		rig = mid;
	}
	rr = lef;

	INT tmp;
	if(p == false) {	
		// 只能产生 (elationTotal - (rr - ll + 1)) 种关系，即去掉训练集中已有的三元组
		tmp = rand_max(id, relation_total - (rr - ll + 1));
	}
	else {
		// 利用概率分布替换 relation
		INT start = r * (relation_total - 1);
		REAL sum = 1;
		bool *record = (bool *)calloc(relation_total - 1, sizeof(bool));
		for (INT i = ll; i <= rr; ++i){
			if (trainRel[i].r > r){
				sum -= prob[start + trainRel[i].r-1];
				record[trainRel[i].r-1] = true;
			}
			else if (trainRel[i].r < r){
				sum -= prob[start + trainRel[i].r];
				record[trainRel[i].r] = true;
			}
		}		
		REAL *prob_tmp = (REAL *)calloc(relation_total-(rr-ll+1), sizeof(REAL));
		INT cnt = 0;
		REAL rec = 0;
		for (INT i = start; i < start + relation_total - 1; ++i) {
			if (record[i-start])
				continue;
			rec += prob[i] / sum;
			prob_tmp[cnt++] = rec;
		}
		REAL m = rand_max(id, 10000) / 10000.0;
		lef = 0;
		rig = cnt - 1;
		while (lef < rig) {
			mid = (lef + rig) >> 1;
			if (prob_tmp[mid] < m) 
				lef = mid + 1; 
			else
				rig = mid;
		}
		tmp = rig;
		free(prob_tmp);
		free(record);
	}

	// 第一种：tmp 小于第一个 t 对应的 relation
	if (tmp < trainRel[ll].r) return tmp;

	// 第二种：tmp 大于最后一个 t 对应的 relation
	if (tmp > trainRel[rr].r - rr + ll - 1) return tmp + rr - ll + 1;
	
	// 第三种：由于 (>= -> rig), (lef + 1 < rig), (tmp + lef - ll + 1)
	// 因此最终返回取值为 (train_head[lef].t, train_head[rig].t) 的 tail
	lef = ll, rig = rr + 1;
	while (lef + 1 < rig) {
		mid = (lef + rig) >> 1;
		// 类似 tmp > trainRel[rr].r - rr + ll - 1
		if (trainRel[mid].r - mid + ll - 1 < tmp)
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
    INT rig = tripleTotal - 1;
    INT mid;
    while (lef + 1 < rig) {
        INT mid = (lef + rig) >> 1;
        if ((tripleList[mid]. h < h) || (tripleList[mid]. h == h && tripleList[mid]. r < r) || (tripleList[mid]. h == h && tripleList[mid]. r == r && tripleList[mid]. t < t)) lef = mid; else rig = mid;
    }
    if (tripleList[lef].h == h && tripleList[lef].r == r && tripleList[lef].t == t) return true;
    if (tripleList[rig].h == h && tripleList[rig].r == r && tripleList[rig].t == t) return true;
	// std::cout << "mid = " << mid << std::endl;
    return false;
}

// 利用 type_constrain.txt 构建负三元组
INT corrupt(INT h, INT r){
	INT ll = tail_lef[r];
	INT rr = tail_rig[r];
	INT loop = 0;
	INT t;
	while(true) {
		t = tail_type[rand(ll, rr)];
		if (not _find(h, t, r)) {
			return t;
		} else {
			loop ++;
			if (loop >= 1000) {
				return corrupt_head(0, h, r);
			}
		} 
	}
}
#endif

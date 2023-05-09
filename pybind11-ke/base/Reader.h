// pybind11-ke/base/Reader.h
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 9, 2023
// 
// 该头文件从数据集中读取三元组.

#ifndef READER_H
#define READER_H
#include "Setting.h"
#include "Triple.h"
#include <cstdlib>
#include <algorithm>
#include <iostream>
#include <cmath>

INT *freqRel, *freqEnt;
INT *lefHead, *rigHead;
INT *lefTail, *rigTail;
INT *lefRel, *rigRel;
REAL *left_mean, *right_mean;
REAL *prob;

Triple *trainList;
Triple *trainHead;
Triple *trainTail;
Triple *trainRel;

INT *testLef, *testRig;
INT *validLef, *validRig;

extern "C"
void importProb(REAL temp){
    if (prob != NULL)
        free(prob);
    FILE *fin;
    fin = fopen((inPath + "kl_prob.txt").c_str(), "r");
    printf("Current temperature:%f\n", temp);
    prob = (REAL *)calloc(relationTotal * (relationTotal - 1), sizeof(REAL));
    INT tmp;
    for (INT i = 0; i < relationTotal * (relationTotal - 1); ++i){
        tmp = fscanf(fin, "%f", &prob[i]);
    }
    REAL sum = 0.0;
    for (INT i = 0; i < relationTotal; ++i) {
        for (INT j = 0; j < relationTotal-1; ++j){
            REAL tmp = exp(-prob[i * (relationTotal - 1) + j] / temp);
            sum += tmp;
            prob[i * (relationTotal - 1) + j] = tmp;
        }
        for (INT j = 0; j < relationTotal-1; ++j){
            prob[i*(relationTotal-1)+j] /= sum;
        }
        sum = 0;
    }
    fclose(fin);
}

// 读取训练集
extern "C"
void importTrainFiles() {

	printf("The toolkit is importing datasets.\n");
	FILE *fin;
	int tmp;

    // 读取关系的个数
    // inPath: defined in Setting.h
    // rel_file: defined in Setting.h
    if (rel_file == "")
	    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    else
        fin = fopen(rel_file.c_str(), "r");
    // relation2id.txt 第一行是关系的个数
	tmp = fscanf(fin, "%ld", &relationTotal);
	printf("The total of relations is %ld.\n", relationTotal);
	fclose(fin);

    // 读取实体的个数
    // ent_file: defined in Setting.h
    if (ent_file == "")
        fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    else
        fin = fopen(ent_file.c_str(), "r");
    // entity2id.txt 第一行是实体的个数
	tmp = fscanf(fin, "%ld", &entityTotal);
	printf("The total of entities is %ld.\n", entityTotal);
	fclose(fin);

    // 读取训练数据集
    // train_file: defined in Setting.h
    if (train_file == "")
        fin = fopen((inPath + "train2id.txt").c_str(), "r");
    else
        fin = fopen(train_file.c_str(), "r");
    // train2id.txt 第一行是三元组的个数
	tmp = fscanf(fin, "%ld", &trainTotal);
    // trainList: 保存训练集中的三元组集合
	trainList = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainHead = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainTail = (Triple *)calloc(trainTotal, sizeof(Triple));
	trainRel = (Triple *)calloc(trainTotal, sizeof(Triple));
    // freqRel, freqEnt: 元素值被初始化为 0.
	freqRel = (INT *)calloc(relationTotal, sizeof(INT));
	freqEnt = (INT *)calloc(entityTotal, sizeof(INT));
    // 读取训练集三元组集合, 保存在 trainList
	for (INT i = 0; i < trainTotal; i++) {
		tmp = fscanf(fin, "%ld", &trainList[i].h);
		tmp = fscanf(fin, "%ld", &trainList[i].t);
		tmp = fscanf(fin, "%ld", &trainList[i].r);
	}
	fclose(fin);
    // 对 trainList 中的三元组排序 (比较顺序: h, r, t).
	std::sort(trainList, trainList + trainTotal, Triple::cmp_head);
    // tmp: 保存训练集三元组的个数
	tmp = trainTotal; trainTotal = 1;
	trainHead[0] = trainTail[0] = trainRel[0] = trainList[0];
    // freqEnt: 保存实体在训练集中出现的总数
    // freqRel: 保存关系在训练集中出现的总数
	freqEnt[trainList[0].t] += 1;
	freqEnt[trainList[0].h] += 1;
	freqRel[trainList[0].r] += 1;
    // 对训练集中的三元组去重
	for (INT i = 1; i < tmp; i++)
		if (trainList[i].h != trainList[i - 1].h || trainList[i].r != trainList[i - 1].r || trainList[i].t != trainList[i - 1].t) {
			trainHead[trainTotal] = trainTail[trainTotal] = trainRel[trainTotal] = trainList[trainTotal] = trainList[i];
			trainTotal++;
			freqEnt[trainList[i].t]++;
			freqEnt[trainList[i].h]++;
			freqRel[trainList[i].r]++;
		}

    // trainHead: 以 h, r, t 排序
    // trainTail: 以 t, r, h 排序
    // trainRel: 以 h, t, r 排序
	std::sort(trainHead, trainHead + trainTotal, Triple::cmp_head);
	std::sort(trainTail, trainTail + trainTotal, Triple::cmp_tail);
	std::sort(trainRel, trainRel + trainTotal, Triple::cmp_rel);
	printf("The total of train triples is %ld.\n", trainTotal);

	lefHead = (INT *)calloc(entityTotal, sizeof(INT));
	rigHead = (INT *)calloc(entityTotal, sizeof(INT));
	lefTail = (INT *)calloc(entityTotal, sizeof(INT));
	rigTail = (INT *)calloc(entityTotal, sizeof(INT));
	lefRel = (INT *)calloc(entityTotal, sizeof(INT));
	rigRel = (INT *)calloc(entityTotal, sizeof(INT));
    // rigHead, rigTail, rigRel 初始化为 -1
	memset(rigHead, -1, sizeof(INT)*entityTotal);
	memset(rigTail, -1, sizeof(INT)*entityTotal);
	memset(rigRel, -1, sizeof(INT)*entityTotal);
	for (INT i = 1; i < trainTotal; i++) {
        // lefTail (entityTotal): 存储每种实体 (tail) 在 trainTail 中第一次出现的位置
        // rigTail (entityTotal): 存储每种实体 (tail) 在 trainTail 中最后一次出现的位置
		if (trainTail[i].t != trainTail[i - 1].t) {
			rigTail[trainTail[i - 1].t] = i - 1;
			lefTail[trainTail[i].t] = i;
		}
        // lefHead (entityTotal): 存储每种实体 (head) 在 trainHead 中第一次出现的位置
        // rigHead (entityTotal): 存储每种实体 (head) 在 trainHead 中最后一次出现的位置
		if (trainHead[i].h != trainHead[i - 1].h) {
			rigHead[trainHead[i - 1].h] = i - 1;
			lefHead[trainHead[i].h] = i;
		}
        // lefRel (entityTotal): 存储每种实体 (head) 在 trainRel 中第一次出现的位置
        // rigRel (entityTotal): 存储每种实体 (head) 在 trainRel 中最后一次出现的位置
		if (trainRel[i].h != trainRel[i - 1].h) {
			rigRel[trainRel[i - 1].h] = i - 1;
			lefRel[trainRel[i].h] = i;
		}
	}
	lefHead[trainHead[0].h] = 0;
	rigHead[trainHead[trainTotal - 1].h] = trainTotal - 1;
	lefTail[trainTail[0].t] = 0;
	rigTail[trainTail[trainTotal - 1].t] = trainTotal - 1;
	lefRel[trainRel[0].h] = 0;
	rigRel[trainRel[trainTotal - 1].h] = trainTotal - 1;

	// 获得 left_mean、right_mean，为 train_mode 中的 bern_flag 做准备
	// 在训练过程中，我们能够构建负三元组进行负采样
	// bern 算法能根据特定关系的 head 和 tail 种类的比值，选择构建适当的负三元组
	// train_mode 中的 bern_flag: pr = left_mean / (left_mean + right_mean)
	// 因此为训练而构建的负三元组比 = tail / (tail + head)
	left_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	right_mean = (REAL *)calloc(relationTotal,sizeof(REAL));
	for (INT i = 0; i < entityTotal; i++) {
		for (INT j = lefHead[i] + 1; j <= rigHead[i]; j++)
			if (trainHead[j].r != trainHead[j - 1].r)
				left_mean[trainHead[j].r] += 1.0;
		if (lefHead[i] <= rigHead[i])
			left_mean[trainHead[lefHead[i]].r] += 1.0;
		for (INT j = lefTail[i] + 1; j <= rigTail[i]; j++)
			if (trainTail[j].r != trainTail[j - 1].r)
				right_mean[trainTail[j].r] += 1.0;
		if (lefTail[i] <= rigTail[i])
			right_mean[trainTail[lefTail[i]].r] += 1.0;
	}
	for (INT i = 0; i < relationTotal; i++) {
		left_mean[i] = freqRel[i] / left_mean[i];
		right_mean[i] = freqRel[i] / right_mean[i];
	}
}

Triple *testList;
Triple *validList;
Triple *tripleList;

// 读取测试集
extern "C"
void importTestFiles() {
    FILE *fin;
    INT tmp;

    // 读取关系的个数
    // inPath: defined in Setting.h
    // rel_file: defined in Setting.h
    if (rel_file == "")
	    fin = fopen((inPath + "relation2id.txt").c_str(), "r");
    else
        fin = fopen(rel_file.c_str(), "r");
    // relation2id.txt 第一行是关系的个数
    tmp = fscanf(fin, "%ld", &relationTotal);
    fclose(fin);

    // 读取实体的个数
    // ent_file: defined in Setting.h
    if (ent_file == "")
        fin = fopen((inPath + "entity2id.txt").c_str(), "r");
    else
        fin = fopen(ent_file.c_str(), "r");
    // entity2id.txt 第一行是实体的个数
    tmp = fscanf(fin, "%ld", &entityTotal);
    fclose(fin);

    // train_file: defined in Setting.h
    // test_file: defined in Setting.h
    // valid_file: defined in Setting.h
    FILE* f_kb1, * f_kb2, * f_kb3;
    if (train_file == "")
        f_kb2 = fopen((inPath + "train2id.txt").c_str(), "r");
    else
        f_kb2 = fopen(train_file.c_str(), "r");
    if (test_file == "")
        f_kb1 = fopen((inPath + "test2id.txt").c_str(), "r");
    else
        f_kb1 = fopen(test_file.c_str(), "r");
    if (valid_file == "")
        f_kb3 = fopen((inPath + "valid2id.txt").c_str(), "r");
    else
        f_kb3 = fopen(valid_file.c_str(), "r");
    // train2id.txt 第一行是三元组的个数
    // test2id.txt 第一行是三元组的个数
    // valid2id.txt 第一行是三元组的个数
    tmp = fscanf(f_kb1, "%ld", &testTotal);
    tmp = fscanf(f_kb2, "%ld", &trainTotal);
    tmp = fscanf(f_kb3, "%ld", &validTotal);
    // tripleTotal: 数据集三元组个数
    tripleTotal = testTotal + trainTotal + validTotal;
    testList = (Triple *)calloc(testTotal, sizeof(Triple));
    validList = (Triple *)calloc(validTotal, sizeof(Triple));
    tripleList = (Triple *)calloc(tripleTotal, sizeof(Triple));
    // 读取测试集三元组
    for (INT i = 0; i < testTotal; i++) {
        tmp = fscanf(f_kb1, "%ld", &testList[i].h);
        tmp = fscanf(f_kb1, "%ld", &testList[i].t);
        tmp = fscanf(f_kb1, "%ld", &testList[i].r);
        tripleList[i] = testList[i];
    }
    // 读取训练集三元组
    for (INT i = 0; i < trainTotal; i++) {
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].h);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].t);
        tmp = fscanf(f_kb2, "%ld", &tripleList[i + testTotal].r);
    }
    // 读取验证集三元组
    for (INT i = 0; i < validTotal; i++) {
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].h);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].t);
        tmp = fscanf(f_kb3, "%ld", &tripleList[i + testTotal + trainTotal].r);
        validList[i] = tripleList[i + testTotal + trainTotal];
    }
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);

    // tripleList: 以 h, r, t 排序
    // testList: 以 r, h, t 排序
    // validList: 以 r, h, t 排序
    std::sort(tripleList, tripleList + tripleTotal, Triple::cmp_head);
    std::sort(testList, testList + testTotal, Triple::cmp_rel2);
    std::sort(validList, validList + validTotal, Triple::cmp_rel2);
    printf("The total of test triples is %ld.\n", testTotal);
    printf("The total of valid triples is %ld.\n", validTotal);

    testLef = (INT *)calloc(relationTotal, sizeof(INT));
    testRig = (INT *)calloc(relationTotal, sizeof(INT));
    // testLef, testRig 初始化为 -1
    memset(testLef, -1, sizeof(INT) * relationTotal);
    memset(testRig, -1, sizeof(INT) * relationTotal);
    for (INT i = 1; i < testTotal; i++) {
	if (testList[i].r != testList[i-1].r) {
        // testLef (relationTotal): 存储每种实体在 testList 中第一次出现的位置
        // testRig (relationTotal): 存储每种实体在 testList 中最后一次出现的位置
	    testRig[testList[i-1].r] = i - 1;
	    testLef[testList[i].r] = i;
	}
    }
    testLef[testList[0].r] = 0;
    testRig[testList[testTotal - 1].r] = testTotal - 1;

    validLef = (INT *)calloc(relationTotal, sizeof(INT));
    validRig = (INT *)calloc(relationTotal, sizeof(INT));
    // validLef, validRig 初始化为 -1
    memset(validLef, -1, sizeof(INT)*relationTotal);
    memset(validRig, -1, sizeof(INT)*relationTotal);
    for (INT i = 1; i < validTotal; i++) {
	if (validList[i].r != validList[i-1].r) {
        // validLef (relationTotal): 存储每种实体在 validList 中第一次出现的位置
        // validRig (relationTotal): 存储每种实体在 validList 中最后一次出现的位置
	    validRig[validList[i-1].r] = i - 1;
	    validLef[validList[i].r] = i;
	}
    }
    validLef[validList[0].r] = 0;
    validRig[validList[validTotal - 1].r] = validTotal - 1;
}

// head_lef: 记录各个关系的 head 类型在 head_type 中第一次出现的位置
// head_rig: 记录各个关系的 head 类型在 head_type 中最后一次出现的后一个位置
// tail_lef: 记录各个关系的 tail 类型在 tail_type 中第一次出现的位置
// tail_rig: 记录各个关系的 tail 类型在 tail_type 中最后一次出现的后一个位置
INT* head_lef;
INT* head_rig;
INT* tail_lef;
INT* tail_rig;
// head_type: 存储各个关系的 head 类型, 各个关系的 head 类型独立地以升序排列
// tail_type: 存储各个关系的 tail 类型, 各个关系的 tail 类型独立地以升序排列
INT* head_type;
INT* tail_type;

// 读取 type_constrain.txt
// type_constrain.txt: 类型约束文件, 第一行是关系的个数
// 下面的行是每个关系的类型限制 (训练集、验证集、测试集中每个关系存在的 head 和 tail 的类型)
// 每个关系有两行：
// 第一行：`id of relation` `Number of head types` `head1` `head2` ...
// 第二行: `id of relation` `number of tail types` `tail1` `tail2` ...
//
// For example, the relation with id 1200 has 4 types of head entities, which are 3123, 1034, 58 and 5733
// The relation with id 1200 has 4 types of tail entities, which are 12123, 4388, 11087 and 11088
// 1200	4	3123	1034	58	5733
// 1200	4	12123	4388	11087	11088
extern "C"
void importTypeFiles() {

    head_lef = (INT *)calloc(relationTotal, sizeof(INT));
    head_rig = (INT *)calloc(relationTotal, sizeof(INT));
    tail_lef = (INT *)calloc(relationTotal, sizeof(INT));
    tail_rig = (INT *)calloc(relationTotal, sizeof(INT));
    // 统计 total_lef, total_rig
    INT total_lef = 0;
    INT total_rig = 0;
    FILE* f_type = fopen((inPath + "type_constrain.txt").c_str(),"r");
    INT tmp;
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld %ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_lef++;
        }
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_rig++;
        }
    }
    fclose(f_type);
    head_type = (INT *)calloc(total_lef, sizeof(INT)); 
    tail_type = (INT *)calloc(total_rig, sizeof(INT));
    // 读取 type_constrain.txt
    total_lef = 0;
    total_rig = 0;
    f_type = fopen((inPath + "type_constrain.txt").c_str(),"r");
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relationTotal; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        head_lef[rel] = total_lef;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &head_type[total_lef]);
            total_lef++;
        }
        head_rig[rel] = total_lef;
        std::sort(head_type + head_lef[rel], head_type + head_rig[rel]);
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        tail_lef[rel] = total_rig;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tail_type[total_rig]);
            total_rig++;
        }
        tail_rig[rel] = total_rig;
        std::sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
    }
    fclose(f_type);
}


#endif
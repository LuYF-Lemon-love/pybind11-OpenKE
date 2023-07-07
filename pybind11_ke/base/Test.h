// pybind11-ke/base/Setting.h
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 23, 2023
// 
// 该头文件进行验证模型.

#ifndef TEST_H
#define TEST_H
#include "Setting.h"
#include "Reader.h"
#include "Corrupt.h"

/*=====================================================================================
link prediction
======================================================================================*/
INT last_head = 0;
INT last_tail = 0;
INT lastRel = 0;
REAL l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l10_tot = 0, r10_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
REAL l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l10_filter_tot = 0, r10_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;
REAL rel3_tot = 0, rel3_filter_tot = 0, rel_filter_tot = 0, rel_filter_rank = 0, rel_rank = 0, rel_filter_reci_rank = 0, rel_reci_rank = 0, rel_tot = 0, rel1_tot = 0, rel1_filter_tot = 0;

REAL l1_filter_tot_constrain = 0, l1_tot_constrain = 0, r1_tot_constrain = 0, r1_filter_tot_constrain = 0, l10_tot_constrain = 0, r10_tot_constrain = 0, l_filter_rank_constrain = 0, l_rank_constrain = 0, l_filter_reci_rank_constrain = 0, l_reci_rank_constrain = 0;
REAL l3_filter_tot_constrain = 0, l3_tot_constrain = 0, r3_tot_constrain = 0, r3_filter_tot_constrain = 0, l10_filter_tot_constrain = 0, r10_filter_tot_constrain = 0, r_filter_rank_constrain = 0, r_rank_constrain = 0, r_filter_reci_rank_constrain = 0, r_reci_rank_constrain = 0;
REAL hit1, hit3, hit10, mr, mrr;
REAL hit1TC, hit3TC, hit10TC, mrTC, mrrTC;

void init_test() {
    last_head = 0;
    last_tail = 0;
    lastRel = 0;
    l1_filter_tot = 0, l1_tot = 0, r1_tot = 0, r1_filter_tot = 0, l10_tot = 0, r10_tot = 0, l_filter_rank = 0, l_rank = 0, l_filter_reci_rank = 0, l_reci_rank = 0;
    l3_filter_tot = 0, l3_tot = 0, r3_tot = 0, r3_filter_tot = 0, l10_filter_tot = 0, r10_filter_tot = 0, r_filter_rank = 0, r_rank = 0, r_filter_reci_rank = 0, r_reci_rank = 0;
    rel3_tot = 0, rel3_filter_tot = 0, rel_filter_tot = 0, rel_filter_rank = 0, rel_rank = 0, rel_filter_reci_rank = 0, rel_reci_rank = 0, rel_tot = 0, rel1_tot = 0, rel1_filter_tot = 0;

    l1_filter_tot_constrain = 0, l1_tot_constrain = 0, r1_tot_constrain = 0, r1_filter_tot_constrain = 0, l10_tot_constrain = 0, r10_tot_constrain = 0, l_filter_rank_constrain = 0, l_rank_constrain = 0, l_filter_reci_rank_constrain = 0, l_reci_rank_constrain = 0;
    l3_filter_tot_constrain = 0, l3_tot_constrain = 0, r3_tot_constrain = 0, r3_filter_tot_constrain = 0, l10_filter_tot_constrain = 0, r10_filter_tot_constrain = 0, r_filter_rank_constrain = 0, r_rank_constrain = 0, r_filter_reci_rank_constrain = 0, r_reci_rank_constrain = 0;
}

// 对于测试集中的给定三元组, 用所有实体替换 head, 返回所有三元组.
void get_head_batch(py::array_t<INT> ph_py, py::array_t<INT> pt_py, py::array_t<INT> pr_py) {
    auto ph = ph_py.mutable_unchecked<1>();
	auto pt = pt_py.mutable_unchecked<1>();
	auto pr = pr_py.mutable_unchecked<1>();
    for (INT i = 0; i < entity_total; i++) {
        ph(i) = i;
        pt(i) = test_list[last_head].t;
        pr(i) = test_list[last_head].r;
    }
    last_head++;
}

// 对于测试集中的给定三元组, 用所有实体替换 tail, 返回所有三元组.
void get_tail_batch(py::array_t<INT> ph_py, py::array_t<INT> pt_py, py::array_t<INT> pr_py) {
    auto ph = ph_py.mutable_unchecked<1>();
	auto pt = pt_py.mutable_unchecked<1>();
	auto pr = pr_py.mutable_unchecked<1>();
    for (INT i = 0; i < entity_total; i++) {
        ph(i) = test_list[last_tail].h;
        pt(i) = i;
        pr(i) = test_list[last_tail].r;
    }
    last_tail++;
}

// 对于测试集中的给定三元组, 用所有实体替换 relation, 返回所有三元组.
extern "C"
void getRelBatch(INT *ph, INT *pt, INT *pr) {
    for (INT i = 0; i < relation_total; i++) {
        ph[i] = test_list[lastRel].h;
        pt[i] = test_list[lastRel].t;
        pr[i] = i;
    }
}

// 替换 head, 评估 head 的 rank.
void test_head(py::array_t<REAL> con_py, INT last_head, bool type_constrain = false) {

    INT h = test_list[last_head].h;
    INT t = test_list[last_head].t;
    INT r = test_list[last_head].r;

    // lef: 记录关系 r 的 head 类型在 head_type 中第一次出现的位置
	// rig: 记录关系 r 的 head 类型在 head_type 中最后一次出现的后一个位置
    INT lef, rig;
    if (type_constrain) {
        lef = head_lef[r];
        rig = head_rig[r];
    }
    // minimal: 正确三元组的 score
    auto con = con_py.mutable_unchecked<1>();
    // REAL minimal = con[h];
    REAL minimal = con(h);

	// l_s: 记录能量 (d(h + l, t)) 小于测试三元组的 (替换 head) 负三元组个数
	// l_filter_s: 记录能量 (d(h + l, t)) 小于测试三元组的 (替换 head) 负三元组个数, 且负三元组不在数据集中
	// l_s_constrain: 记录能量 (d(h + l, t)) 小于测试三元组的 (通过 type_constrain.txt 替换 head 构造负三元组) 负三元组个数
	// l_filter_s_constrain: 记录能量 (d(h + l, t)) 小于测试三元组的 (通过 type_constrain.txt 替换 head 构造负三元组) 负三元组个数, 且负三元组不在数据集中
    INT l_s = 0;
    INT l_filter_s = 0;
    INT l_s_constrain = 0;
    INT l_filter_s_constrain = 0;

    for (INT j = 0; j < entity_total; j++) {
        // 替换 head
        if (j != h) {
            REAL value = con(j);
            if (value < minimal) {
                l_s += 1;
                if (not _find(j, t, r))
                    l_filter_s += 1;
            }
            if (type_constrain) {
                while (lef < rig && head_type[lef] < j) lef++;
                if (lef < rig && j == head_type[lef]) {
                    if (value < minimal) {
                        l_s_constrain += 1;
                        if (not _find(j, t, r)) {
                            l_filter_s_constrain += 1;
                        }
                    }  
                }
            }
        }
    }

    if (l_s < 1) l1_tot += 1;
    if (l_filter_s < 1) l1_filter_tot += 1;
    if (l_s < 3) l3_tot += 1;
    if (l_filter_s < 3) l3_filter_tot += 1;
    if (l_s < 10) l10_tot += 1;
    if (l_filter_s < 10) l10_filter_tot += 1;

    l_rank += (l_s + 1);
    l_filter_rank += (l_filter_s + 1);
    l_reci_rank += 1.0 / (l_s + 1);
    l_filter_reci_rank += 1.0 / (l_filter_s + 1);

    if (type_constrain) {
        if (l_s_constrain < 1) l1_tot_constrain += 1;
        if (l_filter_s_constrain < 1) l1_filter_tot_constrain += 1;
        if (l_s_constrain < 3) l3_tot_constrain += 1;
        if (l_filter_s_constrain < 3) l3_filter_tot_constrain += 1;
        if (l_s_constrain < 10) l10_tot_constrain += 1;
        if (l_filter_s_constrain < 10) l10_filter_tot_constrain += 1;

        l_rank_constrain += (l_s_constrain + 1);
        l_filter_rank_constrain += (l_filter_s_constrain + 1);
        l_reci_rank_constrain += 1.0/(l_s_constrain + 1);
        l_filter_reci_rank_constrain += 1.0/(l_filter_s_constrain + 1);
    }
}

// 替换 tail, 评估 tail 的 rank.
void test_tail(py::array_t<REAL> con_py, INT last_tail, bool type_constrain = false) {

    INT h = test_list[last_tail].h;
    INT t = test_list[last_tail].t;
    INT r = test_list[last_tail].r;

    // lef: 记录关系 r 的 tail 类型在 tail_type 中第一次出现的位置
	// rig: 记录关系 r 的 tail 类型在 tail_type 中最后一次出现的后一个位置
    INT lef, rig;
    if (type_constrain) {
        lef = tail_lef[r];
        rig = tail_rig[r];
    }
    // minimal: 正确三元组的 score
    auto con = con_py.mutable_unchecked<1>();
    // REAL minimal = con[t];
    REAL minimal = con(t);
    // r_s: 记录能量 (d(h + l, t)) 小于测试三元组的 (替换 tail) 负三元组个数
	// r_filter_s: 记录能量 (d(h + l, t)) 小于测试三元组的 (替换 tail) 负三元组个数, 且负三元组不在数据集中
    // r_s_constrain: 记录能量 (d(h + l, t)) 小于测试三元组的 (通过 type_constrain.txt 替换 tail 构造负三元组) 负三元组个数
	// r_filter_s_constrain: 记录能量 (d(h + l, t)) 小于测试三元组的 (通过 type_constrain.txt 替换 tail 构造负三元组) 负三元组个数, 且负三元组不在数据集中
    INT r_s = 0;
    INT r_filter_s = 0;
    INT r_s_constrain = 0;
    INT r_filter_s_constrain = 0;
    for (INT j = 0; j < entity_total; j++) {
        // 替换 tail
        if (j != t) {
            REAL value = con(j);
            if (value < minimal) {
                r_s += 1;
                if (not _find(h, j, r))
                    r_filter_s += 1;
            }
            if (type_constrain) {
                while (lef < rig && tail_type[lef] < j) lef++;
                if (lef < rig && j == tail_type[lef]) {
                        if (value < minimal) {
                            r_s_constrain += 1;
                            if (not _find(h, j ,r)) {
                                r_filter_s_constrain += 1;
                            }
                        }
                }
            }
        }
        
    }

    if (r_s < 1) r1_tot += 1;
    if (r_filter_s < 1) r1_filter_tot += 1;
    if (r_s < 3) r3_tot += 1;
    if (r_filter_s < 3) r3_filter_tot += 1;
    if (r_s < 10) r10_tot += 1;
    if (r_filter_s < 10) r10_filter_tot += 1;

    r_rank += (r_s + 1);
    r_filter_rank += (r_filter_s + 1);
    r_reci_rank += 1.0 / (r_s + 1);
    r_filter_reci_rank += 1.0 / (r_filter_s + 1);
    
    if (type_constrain) {
        if (r_s_constrain < 1) r1_tot_constrain += 1;
        if (r_filter_s_constrain < 1) r1_filter_tot_constrain += 1;
        if (r_s_constrain < 3) r3_tot_constrain += 1;
        if (r_filter_s_constrain < 3) r3_filter_tot_constrain += 1;
        if (r_s_constrain < 10) r10_tot_constrain += 1;
        if (r_filter_s_constrain < 10) r10_filter_tot_constrain += 1;

        r_rank_constrain += (r_s_constrain + 1);
        r_filter_rank_constrain += (r_filter_s_constrain + 1);
        r_reci_rank_constrain += 1.0 / (r_s_constrain + 1);
        r_filter_reci_rank_constrain += 1.0 / (r_filter_s_constrain + 1);
    }
}

// 替换 relation, 评估 relation 的 rank.
extern "C"
void testRel(REAL *con) {
    INT h = test_list[lastRel].h;
    INT t = test_list[lastRel].t;
    INT r = test_list[lastRel].r;

    // minimal: 正确三元组的 score
    REAL minimal = con[r];
    // rel_s: 记录能量 (d(h + l, t)) 小于测试三元组的 (替换 relation) 负三元组个数
	// rel_filter_s: 记录能量 (d(h + l, t)) 小于测试三元组的 (替换 relation) 负三元组个数, 且负三元组不在数据集中
    INT rel_s = 0;
    INT rel_filter_s = 0;

    for (INT j = 0; j < relation_total; j++) {
        // 替换 relation
        if (j != r) {
            REAL value = con[j];
            if (value < minimal) {
                rel_s += 1;
                if (not _find(h, t, j))
                    rel_filter_s += 1;
            }
        }
    }

    if (rel_filter_s < 10) rel_filter_tot += 1;
    if (rel_s < 10) rel_tot += 1;
    if (rel_filter_s < 3) rel3_filter_tot += 1;
    if (rel_s < 3) rel3_tot += 1;
    if (rel_filter_s < 1) rel1_filter_tot += 1;
    if (rel_s < 1) rel1_tot += 1;

    rel_filter_rank += (rel_filter_s+1);
    rel_rank += (1+rel_s);
    rel_filter_reci_rank += 1.0/(rel_filter_s+1);
    rel_reci_rank += 1.0/(rel_s+1);

    lastRel++;
}

// 链接预测入口函数
void test_link_prediction(bool type_constrain = false) {
    l_rank /= test_total;
    r_rank /= test_total;
    l_reci_rank /= test_total;
    r_reci_rank /= test_total;

    l1_tot /= test_total;
    l3_tot /= test_total;
    l10_tot /= test_total;

    r1_tot /= test_total;
    r3_tot /= test_total;
    r10_tot /= test_total;

    // with filter
    l_filter_rank /= test_total;
    r_filter_rank /= test_total;
    l_filter_reci_rank /= test_total;
    r_filter_reci_rank /= test_total;

    l1_filter_tot /= test_total;
    l3_filter_tot /= test_total;
    l10_filter_tot /= test_total;

    r1_filter_tot /= test_total;
    r3_filter_tot /= test_total;
    r10_filter_tot /= test_total;

    printf("no type constraint results:\n");

    printf("metric:\t\t\t MRR \t\t MR \t\t hit@1  \t hit@3  \t hit@10 \n");
    printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_reci_rank, l_rank, l1_tot, l3_tot, l10_tot);
    printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_reci_rank, r_rank, r1_tot, r3_tot, r10_tot);
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
            (l_reci_rank+r_reci_rank)/2, (l_rank+r_rank)/2, (l1_tot+r1_tot)/2, (l3_tot+r3_tot)/2, (l10_tot+r10_tot)/2);
    printf("\n");
    printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_filter_reci_rank, l_filter_rank, l1_filter_tot, l3_filter_tot, l10_filter_tot);
    printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_filter_reci_rank, r_filter_rank, r1_filter_tot, r3_filter_tot, r10_filter_tot);
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
            (l_filter_reci_rank+r_filter_reci_rank)/2, (l_filter_rank+r_filter_rank)/2, (l1_filter_tot+r1_filter_tot)/2, (l3_filter_tot+r3_filter_tot)/2, (l10_filter_tot+r10_filter_tot)/2);

    mrr = (l_filter_reci_rank+r_filter_reci_rank) / 2;
    mr = (l_filter_rank+r_filter_rank) / 2;
    hit1 = (l1_filter_tot+r1_filter_tot) / 2;
    hit3 = (l3_filter_tot+r3_filter_tot) / 2;
    hit10 = (l10_filter_tot+r10_filter_tot) / 2;

    if (type_constrain) {
        //type constrain
        l_rank_constrain /= test_total;
        r_rank_constrain /= test_total;
        l_reci_rank_constrain /= test_total;
        r_reci_rank_constrain /= test_total;

        l1_tot_constrain /= test_total;
        l3_tot_constrain /= test_total;
        l10_tot_constrain /= test_total;

        r1_tot_constrain /= test_total;
        r3_tot_constrain /= test_total;
        r10_tot_constrain /= test_total;

        // with filter
        l_filter_rank_constrain /= test_total;
        r_filter_rank_constrain /= test_total;
        l_filter_reci_rank_constrain /= test_total;
        r_filter_reci_rank_constrain /= test_total;

        l1_filter_tot_constrain /= test_total;
        l3_filter_tot_constrain /= test_total;
        l10_filter_tot_constrain /= test_total;

        r1_filter_tot_constrain /= test_total;
        r3_filter_tot_constrain /= test_total;
        r10_filter_tot_constrain /= test_total;

        printf("type constraint results:\n");
        
        printf("metric:\t\t\t MRR \t\t MR \t\t hit@1  \t hit@3  \t hit@10 \n");
        printf("l(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", l_reci_rank_constrain, l_rank_constrain, l1_tot_constrain, l3_tot_constrain, l10_tot_constrain);
        printf("r(raw):\t\t\t %f \t %f \t %f \t %f \t %f \n", r_reci_rank_constrain, r_rank_constrain, r1_tot_constrain, r3_tot_constrain, r10_tot_constrain);
        printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
                (l_reci_rank_constrain+r_reci_rank_constrain)/2, (l_rank_constrain+r_rank_constrain)/2, (l1_tot_constrain+r1_tot_constrain)/2, (l3_tot_constrain+r3_tot_constrain)/2, (l10_tot_constrain+r10_tot_constrain)/2);
        printf("\n");
        printf("l(filter):\t\t %f \t %f \t %f \t %f \t %f \n", l_filter_reci_rank_constrain, l_filter_rank_constrain, l1_filter_tot_constrain, l3_filter_tot_constrain, l10_filter_tot_constrain);
        printf("r(filter):\t\t %f \t %f \t %f \t %f \t %f \n", r_filter_reci_rank_constrain, r_filter_rank_constrain, r1_filter_tot_constrain, r3_filter_tot_constrain, r10_filter_tot_constrain);
        printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
                (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2, (l_filter_rank_constrain+r_filter_rank_constrain)/2, (l1_filter_tot_constrain+r1_filter_tot_constrain)/2, (l3_filter_tot_constrain+r3_filter_tot_constrain)/2, (l10_filter_tot_constrain+r10_filter_tot_constrain)/2);

        mrrTC = (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2;
        mrTC = (l_filter_rank_constrain+r_filter_rank_constrain) / 2;
        hit1TC = (l1_filter_tot_constrain+r1_filter_tot_constrain) / 2;
        hit3TC = (l3_filter_tot_constrain+r3_filter_tot_constrain) / 2;
        hit10TC = (l10_filter_tot_constrain+r10_filter_tot_constrain) / 2;
    }
}

// 链接预测 (relation) 入口函数
extern "C"
void test_relation_prediction() {
    rel_rank /= test_total;
    rel_reci_rank /= test_total;
  
    rel_tot /= test_total;
    rel3_tot /= test_total;
    rel1_tot /= test_total;

    // with filter
    rel_filter_rank /= test_total;
    rel_filter_reci_rank /= test_total;
  
    rel_filter_tot /= test_total;
    rel3_filter_tot /= test_total;
    rel1_filter_tot /= test_total;

    printf("no type constraint results:\n");
    
    printf("metric:\t\t\t MRR \t\t MR \t\t hit@10 \t hit@3  \t hit@1 \n");
    printf("averaged(raw):\t\t %f \t %f \t %f \t %f \t %f \n",
            rel_reci_rank, rel_rank, rel_tot, rel3_tot, rel1_tot);
    printf("\n");
    printf("averaged(filter):\t %f \t %f \t %f \t %f \t %f \n",
            rel_filter_reci_rank, rel_filter_rank, rel_filter_tot, rel3_filter_tot, rel1_filter_tot);
}

REAL get_test_link_Hit1(bool type_constrain = false) {
    if (type_constrain)
        return hit1TC;    
    return hit1;
}

REAL get_test_link_Hit3(bool type_constrain = false) {
    if (type_constrain)
        return hit3TC;
    return hit3;
}

REAL get_test_link_Hit10(bool type_constrain = false) {
    if (type_constrain)
        return hit10TC;
    return hit10;
}

REAL get_test_link_MR(bool type_constrain = false) {
    if (type_constrain)
        return mrTC;
    return mr;
}

REAL get_test_link_MRR(bool type_constrain = false) {
    if (type_constrain)
        return mrrTC;    
    return mrr;
}


/*=====================================================================================
triple classification
======================================================================================*/
Triple *negTestList = NULL;

extern "C"
void getNegTest() {
    if (negTestList == NULL)
        negTestList = (Triple *)calloc(test_total, sizeof(Triple));
    for (INT i = 0; i < test_total; i++) {
        negTestList[i] = test_list[i];
        if (randd(0) % 1000 < 500)
            negTestList[i].t = corrupt_with_head(0, test_list[i].h, test_list[i].r);
        else
            negTestList[i].h = corrupt_with_tail(0, test_list[i].t, test_list[i].r);
    }
}

// 生成分类数据集 (一半是负数据集, 一半是原始的测试集)
extern "C"
void getTestBatch(INT *ph, INT *pt, INT *pr, INT *nh, INT *nt, INT *nr) {
    getNegTest();
    for (INT i = 0; i < test_total; i++) {
        ph[i] = test_list[i].h;
        pt[i] = test_list[i].t;
        pr[i] = test_list[i].r;
        nh[i] = negTestList[i].h;
        nt[i] = negTestList[i].t;
        nr[i] = negTestList[i].r;
    }
}
#endif

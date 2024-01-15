// pybind11-ke/base/Setting.h
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Dec 30, 2023
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
REAL hit1 = 0, hit3 = 0, hit10 = 0, mr = 0, mrr = 0;
REAL hit1TC = 0, hit3TC = 0, hit10TC = 0, mrTC = 0, mrrTC = 0;

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
void get_head_batch(py::array_t<INT> ph_py, py::array_t<INT> pt_py, py::array_t<INT> pr_py, std::string sampling_mode) {
    auto ph = ph_py.mutable_unchecked<1>();
	auto pt = pt_py.mutable_unchecked<1>();
	auto pr = pr_py.mutable_unchecked<1>();
    if (sampling_mode == "link_test") {
        for (INT i = 0; i < entity_total; i++) {
            ph(i) = i;
            pt(i) = test_list.at(last_head).t;
            pr(i) = test_list.at(last_head).r;
        }
    } else if (sampling_mode == "link_valid") {
        for (INT i = 0; i < entity_total; i++) {
            ph(i) = i;
            pt(i) = valid_list.at(last_head).t;
            pr(i) = valid_list.at(last_head).r;
        }
    }
    last_head++;
}

// 对于测试集中的给定三元组, 用所有实体替换 tail, 返回所有三元组.
void get_tail_batch(py::array_t<INT> ph_py, py::array_t<INT> pt_py, py::array_t<INT> pr_py, std::string sampling_mode) {
    auto ph = ph_py.mutable_unchecked<1>();
	auto pt = pt_py.mutable_unchecked<1>();
	auto pr = pr_py.mutable_unchecked<1>();

    if (sampling_mode == "link_test") {
        for (INT i = 0; i < entity_total; i++) {
            ph(i) = test_list.at(last_tail).h;
            pt(i) = i;
            pr(i) = test_list.at(last_tail).r;
        }
    } else if (sampling_mode == "link_valid") {
        for (INT i = 0; i < entity_total; i++) {
            ph(i) = valid_list.at(last_tail).h;
            pt(i) = i;
            pr(i) = valid_list.at(last_tail).r;
        }
    }
    last_tail++;
}

// 替换 head, 评估 head 的 rank.
void test_head(py::array_t<REAL> con_py, bool type_constrain = false, std::string sampling_mode = "link_test") {

    INT h, t, r;

    if (sampling_mode == "link_test") {
        h = test_list.at(last_head - 1).h;
        t = test_list.at(last_head - 1).t;
        r = test_list.at(last_head - 1).r;
    } else if (sampling_mode == "link_valid") {
        h = valid_list.at(last_head - 1).h;
        t = valid_list.at(last_head - 1).t;
        r = valid_list.at(last_head - 1).r;
    }

    // begin: 记录关系 r 的 head 类型在 head_type_rel 中第一次出现的位置
	// end: 记录关系 r 的 head 类型在 head_type_rel 中最后一次出现的后一个位置
    INT begin, end;
    if (type_constrain) {
        begin = begin_head_type.at(r);
        end = end_head_type.at(r);
    }
    // minimal: 正确三元组的 score
    auto con = con_py.mutable_unchecked<1>();
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
                while (begin < end && head_type_rel.at(begin) < j) begin++;
                if (begin < end && j == head_type_rel.at(begin)) {
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
void test_tail(py::array_t<REAL> con_py, bool type_constrain = false, std::string sampling_mode = "link_test") {

    INT h, t, r;

    if (sampling_mode == "link_test") {
        h = test_list.at(last_tail - 1).h;
        t = test_list.at(last_tail - 1).t;
        r = test_list.at(last_tail - 1).r;
    } else if (sampling_mode == "link_valid") {
        h = valid_list.at(last_tail - 1).h;
        t = valid_list.at(last_tail - 1).t;
        r = valid_list.at(last_tail - 1).r;
    }

    // begin: 记录关系 r 的 tail 类型在 tail_type_rel 中第一次出现的位置
	// end: 记录关系 r 的 tail 类型在 tail_type_rel 中最后一次出现的后一个位置
    INT begin, end;
    if (type_constrain) {
        begin = begin_tail_type.at(r);
        end = end_tail_type.at(r);
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
                while (begin < end && tail_type_rel.at(begin) < j) begin++;
                if (begin < end && j == tail_type_rel.at(begin)) {
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

// 链接预测入口函数
void test_link_prediction(bool type_constrain = false, std::string sampling_mode = "link_test") {
    INT total;

    if (sampling_mode == "link_test") {
        total = test_total;
    } else if (sampling_mode == "link_valid") {
        total = valid_total;
    }

    l_rank /= total;
    r_rank /= total;
    l_reci_rank /= total;
    r_reci_rank /= total;

    l1_tot /= total;
    l3_tot /= total;
    l10_tot /= total;

    r1_tot /= total;
    r3_tot /= total;
    r10_tot /= total;

    // with filter
    l_filter_rank /= total;
    r_filter_rank /= total;
    l_filter_reci_rank /= total;
    r_filter_reci_rank /= total;

    l1_filter_tot /= total;
    l3_filter_tot /= total;
    l10_filter_tot /= total;

    r1_filter_tot /= total;
    r3_filter_tot /= total;
    r10_filter_tot /= total;

    std:: cout << "no type constraint results:" << std::endl;

    std::cout << "metric:\t\t\t MRR \t\t MR \t\t hit@1 \t\t hit@3  \t hit@10" << std::endl;
    std::cout << "l(raw):\t\t\t " << l_reci_rank << " \t "
              << l_rank << " \t " << l1_tot << " \t "
              << l3_tot << " \t " << l10_tot << std::endl;
    std::cout << "r(raw):\t\t\t " << r_reci_rank << " \t "
              << r_rank << " \t " << r1_tot << " \t "
              << r3_tot << " \t " << r10_tot << std::endl;
    std::cout << "averaged(raw):\t\t " << (l_reci_rank+r_reci_rank)/2 << " \t "
              << (l_rank+r_rank)/2 << " \t " << (l1_tot+r1_tot)/2 << " \t "
              << (l3_tot+r3_tot)/2 << " \t " << (l10_tot+r10_tot)/2
              << std::endl << std::endl;

    std::cout << "l(filter):\t\t " << l_filter_reci_rank << " \t "
              << l_filter_rank << " \t " << l1_filter_tot << " \t "
              << l3_filter_tot << " \t " << l10_filter_tot << std::endl;
    std::cout << "r(filter):\t\t " << r_filter_reci_rank << " \t "
              << r_filter_rank << " \t " << r1_filter_tot << " \t "
              << r3_filter_tot << " \t " << r10_filter_tot << std::endl;
    std::cout << "averaged(filter):\t " << (l_filter_reci_rank+r_filter_reci_rank)/2 
              << " \t " << (l_filter_rank+r_filter_rank)/2 << " \t "
              << (l1_filter_tot+r1_filter_tot)/2 << " \t "
              << (l3_filter_tot+r3_filter_tot)/2 << " \t "
              << (l10_filter_tot+r10_filter_tot)/2 << std::endl;

    mrr = (l_filter_reci_rank+r_filter_reci_rank) / 2;
    mr = (l_filter_rank+r_filter_rank) / 2;
    hit1 = (l1_filter_tot+r1_filter_tot) / 2;
    hit3 = (l3_filter_tot+r3_filter_tot) / 2;
    hit10 = (l10_filter_tot+r10_filter_tot) / 2;

    if (type_constrain) {
        //type constrain
        l_rank_constrain /= total;
        r_rank_constrain /= total;
        l_reci_rank_constrain /= total;
        r_reci_rank_constrain /= total;

        l1_tot_constrain /= total;
        l3_tot_constrain /= total;
        l10_tot_constrain /= total;

        r1_tot_constrain /= total;
        r3_tot_constrain /= total;
        r10_tot_constrain /= total;

        // with filter
        l_filter_rank_constrain /= total;
        r_filter_rank_constrain /= total;
        l_filter_reci_rank_constrain /= total;
        r_filter_reci_rank_constrain /= total;

        l1_filter_tot_constrain /= total;
        l3_filter_tot_constrain /= total;
        l10_filter_tot_constrain /= total;

        r1_filter_tot_constrain /= total;
        r3_filter_tot_constrain /= total;
        r10_filter_tot_constrain /= total;

        std::cout << "type constraint results:" << std::endl;
        
        std::cout << "metric:\t\t\t MRR \t\t MR \t\t hit@1 \t\t hit@3  \t hit@10" << std::endl;
        std::cout << "l(raw):\t\t\t " << l_reci_rank_constrain << " \t "
                  << l_rank_constrain << " \t " << l1_tot_constrain << " \t "
                  << l3_tot_constrain << " \t " << l10_tot_constrain << std::endl;
        std::cout << "r(raw):\t\t\t " << r_reci_rank_constrain << " \t "
                  << r_rank_constrain << " \t " << r1_tot_constrain << " \t "
                  << r3_tot_constrain << " \t " << r10_tot_constrain << std::endl;
        std::cout << "averaged(raw):\t\t " << (l_reci_rank_constrain+r_reci_rank_constrain)/2 << " \t "
                  << (l_rank_constrain+r_rank_constrain)/2 << " \t "
                  << (l1_tot_constrain+r1_tot_constrain)/2 << " \t "
                  << (l3_tot_constrain+r3_tot_constrain)/2 << " \t "
                  << (l10_tot_constrain+r10_tot_constrain)/2 << std::endl << std::endl;
        
        std::cout << "l(filter):\t\t " << l_filter_reci_rank_constrain << " \t "
                  << l_filter_rank_constrain << " \t " << l1_filter_tot_constrain << " \t "
                  << l3_filter_tot_constrain << " \t " << l10_filter_tot_constrain << std::endl;
        std::cout << "r(filter):\t\t " << r_filter_reci_rank_constrain << " \t "
                  << r_filter_rank_constrain << " \t " << r1_filter_tot_constrain << " \t "
                  << r3_filter_tot_constrain << " \t " << r10_filter_tot_constrain << std::endl;
        std::cout << "averaged(filter):\t "
                  << (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2 << " \t "
                  << (l_filter_rank_constrain+r_filter_rank_constrain)/2 << " \t "
                  << (l1_filter_tot_constrain+r1_filter_tot_constrain)/2 << " \t "
                  << (l3_filter_tot_constrain+r3_filter_tot_constrain)/2 << " \t "
                  << (l10_filter_tot_constrain+r10_filter_tot_constrain)/2 << std::endl;

        mrrTC = (l_filter_reci_rank_constrain+r_filter_reci_rank_constrain)/2;
        mrTC = (l_filter_rank_constrain+r_filter_rank_constrain) / 2;
        hit1TC = (l1_filter_tot_constrain+r1_filter_tot_constrain) / 2;
        hit3TC = (l3_filter_tot_constrain+r3_filter_tot_constrain) / 2;
        hit10TC = (l10_filter_tot_constrain+r10_filter_tot_constrain) / 2;
    }
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
#endif

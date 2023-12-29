// pybind11-ke/base/Reader.h
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Dec 28, 2023
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
#include <fstream>

std::vector<INT> first_head, end_head, first_tail, end_tail, first_rel, end_rel;
std::vector<REAL> hpt, tph;
std::vector<Triple> train_list, train_head, train_tail, train_rel;

// 读取训练集
void read_train_files() {

    std::cout << "The toolkit is importing datasets." << std::endl;
    std::ifstream istrm;
    int tmp;

    // 读取关系的个数
    istrm.open(rel_file, std::ifstream::in);
    istrm >> relation_total;
    istrm.close();
    std::cout << "The total of relations is " << relation_total
        << "." << std::endl;

    // 读取实体的个数
    istrm.open(ent_file, std::ifstream::in);
    istrm >> entity_total;
    istrm.close();
    std::cout << "The total of entities is " << entity_total
        << "." << std::endl;

    // 读取训练数据集
    istrm.open(train_file, std::ifstream::in);
    istrm >> train_total;
    // train_list: 保存训练集中的三元组集合.
    train_list.resize(train_total);
    train_head.resize(train_total);
    train_tail.resize(train_total);
    train_rel.resize(train_total);
    // freq_rel 元素值被初始化为 0.
    std::vector<INT> freq_rel(relation_total, 0);
    // 读取训练集三元组集合, 保存在 train_list.
    for (INT i = 0; i < train_total; i++) {
        istrm >> train_list[i].h >> train_list[i].t >> train_list[i].r;
    }
    istrm.close();
    // 对 train_list 中的三元组排序 (比较顺序: h, r, t).
    std::sort(train_list.begin(), train_list.end(), Triple::cmp_head);
    // tmp: 保存训练集三元组的个数
    tmp = train_total; train_total = 1;
    train_head[0] = train_tail[0] = train_rel[0] = train_list[0];
    // freq_rel: 保存每个关系训练集中三元组的个数
    freq_rel[train_list[0].r] += 1;
    // 对训练集中的三元组去重
    for (INT i = 1; i < tmp; i++)
        if (train_list[i].h != train_list[i - 1].h
            || train_list[i].r != train_list[i - 1].r
            || train_list[i].t != train_list[i - 1].t) {
            train_head[train_total] = train_tail[train_total]
                = train_rel[train_total] = train_list[train_total]
                = train_list[i];
            train_total++;
            freq_rel[train_list[i].r]++;
        }

    // train_head: 以 h, r, t 排序
    // train_tail: 以 t, r, h 排序
    // train_rel: 以 h, t, r 排序
    std::sort(train_head.begin(), train_head.end(), Triple::cmp_head);
    std::sort(train_tail.begin(), train_tail.end(), Triple::cmp_tail);
    std::sort(train_rel.begin(), train_rel.end(), Triple::cmp_rel);
    std::cout << "The total of train triples is " << train_total
        << "." << std::endl;
    
    first_head.resize(entity_total);
    end_head.resize(entity_total);
    first_tail.resize(entity_total);
    end_tail.resize(entity_total);
    first_rel.resize(entity_total);
    end_rel.resize(entity_total);
    for (INT i = 1; i < train_total; i++) {
        // first_head (entity_total): 存储每种实体 (head) 在 train_head 中第一次出现的位置
        // end_head (entity_total): 存储每种实体 (head) 在 train_head 中最后一次出现的位置
        if (train_head[i].h != train_head[i - 1].h) {
            end_head[train_head[i - 1].h] = i - 1;
            first_head[train_head[i].h] = i;
        }
        // first_tail (entity_total): 存储每种实体 (tail) 在 train_tail 中第一次出现的位置
        // end_tail (entity_total): 存储每种实体 (tail) 在 train_tail 中最后一次出现的位置
        if (train_tail[i].t != train_tail[i - 1].t) {
            end_tail[train_tail[i - 1].t] = i - 1;
            first_tail[train_tail[i].t] = i;
        }
        // first_rel (entity_total): 存储每种实体 (head) 在 train_rel 中第一次出现的位置
        // end_rel (entity_total): 存储每种实体 (head) 在 train_rel 中最后一次出现的位置
        if (train_rel[i].h != train_rel[i - 1].h) {
            end_rel[train_rel[i - 1].h] = i - 1;
            first_rel[train_rel[i].h] = i;
        }
    }
    first_head[train_head[0].h] = 0;
    end_head[train_head[train_total - 1].h] = train_total - 1;
    first_tail[train_tail[0].t] = 0;
    end_tail[train_tail[train_total - 1].t] = train_total - 1;
    first_rel[train_rel[0].h] = 0;
    end_rel[train_rel[train_total - 1].h] = train_total - 1;
    
    // 为 bern 负采样做准备
    std::vector<REAL> heads_rel(relation_total, 0.0), tails_rel(relation_total, 0.0);
    hpt.resize(relation_total, 0.0);
    tph.resize(relation_total, 0.0);
    for (INT i = 0; i < entity_total; i++) {
        for (INT j = first_head[i] + 1; j <= end_head[i]; j++)
            if (train_head[j].r != train_head[j - 1].r)
                heads_rel[train_head[j].r] += 1.0;
        if (first_head[i] <= end_head[i])
            heads_rel[train_head[first_head[i]].r] += 1.0;
        for (INT j = first_tail[i] + 1; j <= end_tail[i]; j++)
            if (train_tail[j].r != train_tail[j - 1].r)
                tails_rel[train_tail[j].r] += 1.0;
        if (first_tail[i] <= end_tail[i])
            tails_rel[train_tail[first_tail[i]].r] += 1.0;
    }
    for (INT i = 0; i < relation_total; i++) {
        tph[i] = freq_rel[i] / heads_rel[i];
        hpt[i] = freq_rel[i] / tails_rel[i];
    }
}

Triple *test_list;
Triple *valid_list;
Triple *triple_list;

// 读取测试集
void read_test_files() {
    std::ifstream istrm;

    // 读取关系的个数
    istrm.open(rel_file, std::ifstream::in);
    istrm >> relation_total;
    istrm.close();

    // 读取实体的个数
    istrm.open(ent_file, std::ifstream::in);
    istrm >> entity_total;
    istrm.close();

    // 读取训练集、验证集和测试集
    std::ifstream istrm_test, istrm_train, istrm_valid;
    istrm_test.open(test_file, std::ifstream::in);
    istrm_train.open(train_file, std::ifstream::in);
    istrm_valid.open(valid_file, std::ifstream::in);
    // test2id.txt 第一行是三元组的个数
    // train2id.txt 第一行是三元组的个数
    // valid2id.txt 第一行是三元组的个数
    istrm_test >> test_total;
    istrm_train >> train_total;
    istrm_valid >> valid_total;
    std::cout << "The total of test triples is " << test_total
            << "." << std::endl;
    std::cout << "The total of valid triples is " << valid_total
            << "." << std::endl;
    
    // triple_total: 数据集三元组个数
    triple_total = test_total + train_total + valid_total;
    test_list = (Triple *)calloc(test_total, sizeof(Triple));
    valid_list = (Triple *)calloc(valid_total, sizeof(Triple));
    triple_list = (Triple *)calloc(triple_total, sizeof(Triple));
    // 读取测试集三元组
    for (INT i = 0; i < test_total; i++) {
        istrm_test >> test_list[i].h >> test_list[i].t >> test_list[i].r;
        triple_list[i] = test_list[i];
    }
    // 读取训练集三元组
    for (INT i = 0; i < train_total; i++) {
        istrm_train >> triple_list[i + test_total].h >> triple_list[i + test_total].t
                    >> triple_list[i + test_total].r;
    }
    // 读取验证集三元组
    for (INT i = 0; i < valid_total; i++) {
        istrm_valid >> triple_list[i + test_total + train_total].h 
                    >> triple_list[i + test_total + train_total].t
                    >> triple_list[i + test_total + train_total].r;
        valid_list[i] = triple_list[i + test_total + train_total];
    }
    istrm_test.close();
    istrm_train.close();
    istrm_valid.close();

    // triple_list: 以 h, r, t 排序
    // test_list: 以 r, h, t 排序
    // valid_list: 以 r, h, t 排序
    std::sort(triple_list, triple_list + triple_total, Triple::cmp_head);
    std::sort(test_list, test_list + test_total, Triple::cmp_rel2);
    std::sort(valid_list, valid_list + valid_total, Triple::cmp_rel2);
}

// head_type: 存储各个关系的 head 类型, 各个关系的 head 类型独立地以升序排列
// tail_type: 存储各个关系的 tail 类型, 各个关系的 tail 类型独立地以升序排列
INT* head_type;
INT* tail_type;
// head_lef: 记录各个关系的 head 类型在 head_type 中第一次出现的位置
// head_rig: 记录各个关系的 head 类型在 head_type 中最后一次出现的后一个位置
// tail_lef: 记录各个关系的 tail 类型在 tail_type 中第一次出现的位置
// tail_rig: 记录各个关系的 tail 类型在 tail_type 中最后一次出现的后一个位置
INT* head_lef;
INT* head_rig;
INT* tail_lef;
INT* tail_rig;

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
void read_type_files() {

    head_lef = (INT *)calloc(relation_total, sizeof(INT));
    head_rig = (INT *)calloc(relation_total, sizeof(INT));
    tail_lef = (INT *)calloc(relation_total, sizeof(INT));
    tail_rig = (INT *)calloc(relation_total, sizeof(INT));
    // 统计所有关系头实体类型、尾实体类型的总数
    INT total_head = 0;
    INT total_tail = 0;
    FILE* f_type = fopen((in_path + "type_constrain.txt").c_str(), "r");
    INT tmp;
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relation_total; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld %ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_head++;
        }
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tmp);
            total_tail++;
        }
    }
    fclose(f_type);
    head_type = (INT *)calloc(total_head, sizeof(INT)); 
    tail_type = (INT *)calloc(total_tail, sizeof(INT));
    // 读取 type_constrain.txt
    total_head = 0;
    total_tail = 0;
    f_type = fopen((in_path + "type_constrain.txt").c_str(),"r");
    tmp = fscanf(f_type, "%ld", &tmp);
    for (INT i = 0; i < relation_total; i++) {
        INT rel, tot;
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        head_lef[rel] = total_head;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &head_type[total_head]);
            total_head++;
        }
        head_rig[rel] = total_head;
        std::sort(head_type + head_lef[rel], head_type + head_rig[rel]);
        tmp = fscanf(f_type, "%ld%ld", &rel, &tot);
        tail_lef[rel] = total_tail;
        for (INT j = 0; j < tot; j++) {
            tmp = fscanf(f_type, "%ld", &tail_type[total_tail]);
            total_tail++;
        }
        tail_rig[rel] = total_tail;
        std::sort(tail_type + tail_lef[rel], tail_type + tail_rig[rel]);
    }
    fclose(f_type);
}

#endif
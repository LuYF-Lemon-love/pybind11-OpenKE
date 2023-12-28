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

INT *freq_rel, *freq_ent;
INT *left_head, *right_head;
INT *left_tail, *right_tail;
INT *left_rel, *right_rel;
REAL *left_mean, *right_mean;

Triple *train_list;
Triple *train_head;
Triple *train_tail;
Triple *train_rel;

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
    train_list = (Triple *)calloc(train_total, sizeof(Triple));
    train_head = (Triple *)calloc(train_total, sizeof(Triple));
    train_tail = (Triple *)calloc(train_total, sizeof(Triple));
    train_rel = (Triple *)calloc(train_total, sizeof(Triple));
    // freq_rel, freq_ent: 元素值被初始化为 0.
    freq_rel = (INT *)calloc(relation_total, sizeof(INT));
    freq_ent = (INT *)calloc(entity_total, sizeof(INT));
    // 读取训练集三元组集合, 保存在 train_list.
    for (INT i = 0; i < train_total; i++) {
        istrm >> train_list[i].h >> train_list[i].t >> train_list[i].r;
    }
    istrm.close();
    // 对 train_list 中的三元组排序 (比较顺序: h, r, t).
    std::sort(train_list, train_list + train_total, Triple::cmp_head);
    // tmp: 保存训练集三元组的个数
    tmp = train_total; train_total = 1;
    train_head[0] = train_tail[0] = train_rel[0] = train_list[0];
    // freq_ent: 保存实体在训练集中出现的总数
    // freq_rel: 保存关系在训练集中出现的总数
    freq_ent[train_list[0].t] += 1;
    freq_ent[train_list[0].h] += 1;
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
            freq_ent[train_list[i].t]++;
            freq_ent[train_list[i].h]++;
            freq_rel[train_list[i].r]++;
        }

    // train_head: 以 h, r, t 排序
    // train_tail: 以 t, r, h 排序
    // train_rel: 以 h, t, r 排序
    std::sort(train_head, train_head + train_total, Triple::cmp_head);
    std::sort(train_tail, train_tail + train_total, Triple::cmp_tail);
    std::sort(train_rel, train_rel + train_total, Triple::cmp_rel);
    std::cout << "The total of train triples is " << train_total
        << "." << std::endl;
    
    left_head = (INT *)calloc(entity_total, sizeof(INT));
    right_head = (INT *)calloc(entity_total, sizeof(INT));
    left_tail = (INT *)calloc(entity_total, sizeof(INT));
    right_tail = (INT *)calloc(entity_total, sizeof(INT));
    left_rel = (INT *)calloc(entity_total, sizeof(INT));
    right_rel = (INT *)calloc(entity_total, sizeof(INT));
    for (INT i = 1; i < train_total; i++) {
        // left_head (entity_total): 存储每种实体 (head) 在 train_head 中第一次出现的位置
        // right_head (entity_total): 存储每种实体 (head) 在 train_head 中最后一次出现的位置
        if (train_head[i].h != train_head[i - 1].h) {
            right_head[train_head[i - 1].h] = i - 1;
            left_head[train_head[i].h] = i;
        }
        // left_tail (entity_total): 存储每种实体 (tail) 在 train_tail 中第一次出现的位置
        // right_tail (entity_total): 存储每种实体 (tail) 在 train_tail 中最后一次出现的位置
        if (train_tail[i].t != train_tail[i - 1].t) {
            right_tail[train_tail[i - 1].t] = i - 1;
            left_tail[train_tail[i].t] = i;
        }
        // left_rel (entity_total): 存储每种实体 (head) 在 train_rel 中第一次出现的位置
        // right_rel (entity_total): 存储每种实体 (head) 在 train_rel 中最后一次出现的位置
        if (train_rel[i].h != train_rel[i - 1].h) {
            right_rel[train_rel[i - 1].h] = i - 1;
            left_rel[train_rel[i].h] = i;
        }
    }
    left_head[train_head[0].h] = 0;
    right_head[train_head[train_total - 1].h] = train_total - 1;
    left_tail[train_tail[0].t] = 0;
    right_tail[train_tail[train_total - 1].t] = train_total - 1;
    left_rel[train_rel[0].h] = 0;
    right_rel[train_rel[train_total - 1].h] = train_total - 1;
    
    // 获得 left_mean、right_mean，为 train_mode 中的 bern 做准备
    // 在训练过程中，我们能够构建负三元组进行负采样
    // bern 算法能根据特定关系的 head 和 tail 种类的比值，选择构建适当的负三元组
    // train_mode 中的 bern: pr = left_mean / (left_mean + right_mean)
    // 因此为训练而构建的负三元组比 = tail / (tail + head)
    left_mean = (REAL *)calloc(relation_total, sizeof(REAL));
    right_mean = (REAL *)calloc(relation_total, sizeof(REAL));
    for (INT i = 0; i < entity_total; i++) {
        for (INT j = left_head[i] + 1; j <= right_head[i]; j++)
            if (train_head[j].r != train_head[j - 1].r)
                left_mean[train_head[j].r] += 1.0;
        if (left_head[i] <= right_head[i])
            left_mean[train_head[left_head[i]].r] += 1.0;
        for (INT j = left_tail[i] + 1; j <= right_tail[i]; j++)
            if (train_tail[j].r != train_tail[j - 1].r)
                right_mean[train_tail[j].r] += 1.0;
        if (left_tail[i] <= right_tail[i])
            right_mean[train_tail[left_tail[i]].r] += 1.0;
    }
    for (INT i = 0; i < relation_total; i++) {
        left_mean[i] = freq_rel[i] / left_mean[i];
        right_mean[i] = freq_rel[i] / right_mean[i];
    }
}

Triple *test_list;
Triple *valid_list;
Triple *triple_list;

// 读取测试集
void read_test_files() {
    FILE *fin;
    INT tmp;

    // 读取关系的个数
    // rel_file: 定义于 Setting.h
    fin = fopen(rel_file.c_str(), "r");
    // relation2id.txt 第一行是关系的个数
    tmp = fscanf(fin, "%ld", &relation_total);
    fclose(fin);

    // 读取实体的个数
    // ent_file: 定义于 Setting.h
    fin = fopen(ent_file.c_str(), "r");
    // entity2id.txt 第一行是实体的个数
    tmp = fscanf(fin, "%ld", &entity_total);
    fclose(fin);

    // test_file: 定义于 Setting.h
    // train_file: 定义于 Setting.h
    // valid_file: 定义于 Setting.h
    FILE* f_kb1, * f_kb2, * f_kb3;
    f_kb1 = fopen(test_file.c_str(), "r");
    f_kb2 = fopen(train_file.c_str(), "r");
    f_kb3 = fopen(valid_file.c_str(), "r");
    // test2id.txt 第一行是三元组的个数
    // train2id.txt 第一行是三元组的个数
    // valid2id.txt 第一行是三元组的个数
    tmp = fscanf(f_kb1, "%ld", &test_total);
    tmp = fscanf(f_kb2, "%ld", &train_total);
    tmp = fscanf(f_kb3, "%ld", &valid_total);
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
        tmp = fscanf(f_kb1, "%ld", &test_list[i].h);
        tmp = fscanf(f_kb1, "%ld", &test_list[i].t);
        tmp = fscanf(f_kb1, "%ld", &test_list[i].r);
        triple_list[i] = test_list[i];
    }
    // 读取训练集三元组
    for (INT i = 0; i < train_total; i++) {
        tmp = fscanf(f_kb2, "%ld", &triple_list[i + test_total].h);
        tmp = fscanf(f_kb2, "%ld", &triple_list[i + test_total].t);
        tmp = fscanf(f_kb2, "%ld", &triple_list[i + test_total].r);
    }
    // 读取验证集三元组
    for (INT i = 0; i < valid_total; i++) {
        tmp = fscanf(f_kb3, "%ld", &triple_list[i + test_total + train_total].h);
        tmp = fscanf(f_kb3, "%ld", &triple_list[i + test_total + train_total].t);
        tmp = fscanf(f_kb3, "%ld", &triple_list[i + test_total + train_total].r);
        valid_list[i] = triple_list[i + test_total + train_total];
    }
    fclose(f_kb1);
    fclose(f_kb2);
    fclose(f_kb3);

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
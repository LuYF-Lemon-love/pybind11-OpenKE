// pybind11-ke/base/Setting.h
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on Dec 28, 2023
// 
// 该头文件定义了一些全局配置变量.

#ifndef SETTING_H
#define SETTING_H

// 引用头文件
#include <iostream>
#include <cstring>
#include <cstdio>
#include <string>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <algorithm>
#include <cmath>
#include <random>

// pybind11 的命名空间
namespace py = pybind11;

using INT = long;
using REAL = float;

// 文件路径
// in_path: 数据集目录
// ent_file: entity2id.txt
// rel_file: relation2id.txt
// train_file: train2id.txt
// valid_file: valid2id.txt
// test_file: test2id.txt
std::string in_path = "";
std::string ent_file = "";
std::string rel_file = "";
std::string train_file = "";
std::string valid_file = "";
std::string test_file = "";

// 设置输入文件路径
void set_in_path(std::string path) {
	in_path = std::move(path);
    std::cout << "Input Files Path : "
              << in_path << std::endl;
}

// 设置实体数据文件路径
void set_ent_path(std::string path) {
	ent_file = std::move(path);
    std::cout << "Entity Files Path : "
              << ent_file << std::endl;
}

// 设置关系数据文件路径
void set_rel_path(std::string path) {
	rel_file = std::move(path);
    std::cout << "Relation Files Path : "
              << rel_file << std::endl;
}

// 设置训练集数据文件路径
void set_train_path(std::string path) {
	train_file = std::move(path);
    std::cout << "Training Files Path : "
              << train_file << std::endl;
}

// 设置验证集数据文件路径
void set_valid_path(std::string path) {
	valid_file = std::move(path);
    std::cout << "Valid Files Path : "
              << valid_file << std::endl;
}

// 设置测试集数据文件路径
void set_test_path(std::string path) {
	test_file = std::move(path);
    std::cout << "Test Files Path : "
              << test_file << std::endl;
}

/*
============================================================
*/

// TransH 提出的负采样策略
bool bern = false;

void set_bern(bool flag) {
	bern = flag;
}

/*
============================================================
*/

// 线程数
INT work_threads = 1;

void set_work_threads(INT threads) {
	work_threads = threads;
}

/*
============================================================
*/

// 统计数据
INT relation_total = 0;
INT entity_total = 0;
INT triple_total = 0;
INT test_total = 0;
INT train_total = 0;
INT valid_total = 0;

INT get_entity_total() {
	return entity_total;
}

INT get_relation_total() {
	return relation_total;
}

INT get_train_total() {
	return train_total;
}

INT get_test_total() {
	return test_total;
}

INT get_valid_total() {
	return valid_total;
}

#endif
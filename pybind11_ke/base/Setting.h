// pybind11-ke/base/Setting.h
// 
// git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
// updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 8, 2023
// 
// 该头文件定义了一些全局配置变量.

#ifndef SETTING_H
#define SETTING_H
#define INT long
#define REAL float
#include <iostream>
#include <cstring>
#include <cstdio>
#include <string>
#include <pybind11/pybind11.h> //导入 pybind11
#include <pybind11/numpy.h>
namespace py = pybind11;

// 文件路径
// inPath: 数据集目录
// ent_file: entity2id.txt
// rel_file: relation2id.txt
// train_file: train2id.txt
// valid_file: valid2id.txt
// test_file: test2id.txt
std::string inPath = "../data/FB15K/";
std::string outPath = "../data/FB15K/";
std::string ent_file = "";
std::string rel_file = "";
std::string train_file = "";
std::string valid_file = "";
std::string test_file = "";

// // 设置输入文件路径
// extern "C"
// void setInPath(char *path) {
// 	INT len = strlen(path);
// 	inPath = "";
// 	for (INT i = 0; i < len; i++)
// 		inPath = inPath + path[i];
// 	printf("Input Files Path : %s\n", inPath.c_str());
// }

// 设置输入文件路径
void setInPath(std::string path) {
	inPath = std::move(path);
    std::cout << "Input Files Path : "
              << inPath << std::endl;
}

// 设置输出文件路径
extern "C"
void setOutPath(char *path) {
	INT len = strlen(path);
	outPath = "";
	for (INT i = 0; i < len; i++)
		outPath = outPath + path[i];
	printf("Output Files Path : %s\n", outPath.c_str());
}

// // 设置训练集数据文件路径
// extern "C"
// void setTrainPath(char *path) {
// 	INT len = strlen(path);
// 	train_file = "";
// 	for (INT i = 0; i < len; i++)
// 		train_file = train_file + path[i];
// 	printf("Training Files Path : %s\n", train_file.c_str());
// }

// 设置训练集数据文件路径
void setTrainPath(std::string path) {
	train_file = std::move(path);
    std::cout << "Training Files Path : "
              << train_file << std::endl;
}

// 设置验证集数据文件路径
extern "C"
void setValidPath(char *path) {
	INT len = strlen(path);
	valid_file = "";
	for (INT i = 0; i < len; i++)
		valid_file = valid_file + path[i];
	printf("Valid Files Path : %s\n", valid_file.c_str());
}

// 设置测试集数据文件路径
extern "C"
void setTestPath(char *path) {
	INT len = strlen(path);
	test_file = "";
	for (INT i = 0; i < len; i++)
		test_file = test_file + path[i];
	printf("Test Files Path : %s\n", test_file.c_str());
}

// // 设置实体数据文件路径
// extern "C"
// void setEntPath(char *path) {
// 	INT len = strlen(path);
// 	ent_file = "";
// 	for (INT i = 0; i < len; i++)
// 		ent_file = ent_file + path[i];
// 	printf("Entity Files Path : %s\n", ent_file.c_str());
// }

// 设置实体数据文件路径
void setEntPath(std::string path) {
	ent_file = std::move(path);
    std::cout << "Entity Files Path : "
              << ent_file << std::endl;
}

// // 设置关系数据文件路径
// extern "C"
// void setRelPath(char *path) {
// 	INT len = strlen(path);
// 	rel_file = "";
// 	for (INT i = 0; i < len; i++)
// 		rel_file = rel_file + path[i];
// 	printf("Relation Files Path : %s\n", rel_file.c_str());
// }

// 设置关系数据文件路径
void setRelPath(std::string path) {
	rel_file = std::move(path);
    std::cout << "Relation Files Path : "
              << rel_file << std::endl;
}

/*
============================================================
*/

// 线程数
INT workThreads = 1;

// extern "C"
void setWorkThreads(INT threads) {
	workThreads = threads;
}

extern "C"
INT getWorkThreads() {
	return workThreads;
}

/*
============================================================
*/

// 统计数据
INT relationTotal = 0;
INT entityTotal = 0;
INT tripleTotal = 0;
INT testTotal = 0;
INT trainTotal = 0;
INT validTotal = 0;

// extern "C"
INT getEntityTotal() {
	return entityTotal;
}

// extern "C"
INT getRelationTotal() {
	return relationTotal;
}

extern "C"
INT getTripleTotal() {
	return tripleTotal;
}

// extern "C"
INT getTrainTotal() {
	return trainTotal;
}

// extern "C"
INT getTestTotal() {
	return testTotal;
}

extern "C"
INT getValidTotal() {
	return validTotal;
}
/*
============================================================
*/

// TransH 提出的负采样策略
INT bernFlag = 0;

// extern "C"
// void setBern(INT con) {
// 	bernFlag = con;
// }

void setBern(INT con) {
	bernFlag = con;
}

#endif

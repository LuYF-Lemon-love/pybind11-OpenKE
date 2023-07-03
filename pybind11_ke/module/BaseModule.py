# coding:utf-8
#
# pybind11_ke/module/BaseModule.py
# 
# git pull from OpenKE-PyTorch by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 7, 2023
# updated by LuYF-Lemon-love <luyanfeng_nlp@qq.com> on May 27, 2023
# 
# 该头文件定义了 BaseModule.

"""BaseModule - 所有模块的基类"""

import torch
import torch.nn as nn
import os
import json

class BaseModule(nn.Module):
	""":py:class:`BaseModule` 继承自 :py:class:`torch.nn.Module`，
	并且封装了一些常用功能，如加载和保存模型。"""

	def __init__(self):
		"""创建 BaseModule 对象。"""

		super(BaseModule, self).__init__()

		#: 常数 0
		self.zero_const = nn.Parameter(torch.Tensor([0]))
		self.zero_const.requires_grad = False

		#: 常数 pi
		self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
		self.pi_const.requires_grad = False

	def load_checkpoint(self, path):
		"""加载模型权重。

		:param path: 模型保存的路径
		:type path: str
		"""

		self.load_state_dict(torch.load(os.path.join(path)))
		self.eval()

	def save_checkpoint(self, path):

		"""保存模型权重。

		:param path: 模型保存的路径
		:type path: str
		"""

		if not os.path.exists(os.path.split(path)[0]):
			os.makedirs(os.path.split(path)[0], exist_ok=True)
		torch.save(self.state_dict(), path)

	def load_parameters(self, path):

		"""加载模型权重。

		:param path: 模型保存的路径
		:type path: str
		"""

		f = open(path, "r")
		parameters = json.loads(f.read())
		f.close()
		for i in parameters:
			parameters[i] = torch.Tensor(parameters[i])
		self.load_state_dict(parameters, strict = False)
		self.eval()

	def save_parameters(self, path):

		"""保存模型权重。

		:param path: 模型保存的路径
		:type path: str
		"""

		f = open(path, "w")
		f.write(json.dumps(self.get_parameters("list")))
		f.close()

	def get_parameters(self, mode = "numpy", param_dict = None):

		"""获得模型权重。

		:param mode: 模型保存的格式，可以选择 ``numpy``、``list`` 和 ``Tensor``。
		:type path: str
		:param param_dict: 可以选择从哪里获得模型权重。
		:type param_dict: dict
		:returns: 模型权重字典。
		:rtype: dict
		"""

		all_param_dict = self.state_dict()
		if param_dict == None:
			param_dict = all_param_dict.keys()
		res = {}
		for param in param_dict:
			if mode == "numpy":
				res[param] = all_param_dict[param].cpu().numpy()
			elif mode == "list":
				res[param] = all_param_dict[param].cpu().numpy().tolist()
			else:
				res[param] = all_param_dict[param]
		return res

	def set_parameters(self, parameters):
		
		"""加载模型权重。

		:param parameters: 模型权重字典。
		:type parameters: dict
		"""

		for i in parameters:
			parameters[i] = torch.Tensor(parameters[i])
		self.load_state_dict(parameters, strict = False)
		self.eval()
# OOD-Detector: 基于CLIP和证据理论的分布外检测框架

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-blue.svg)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/Lightning-2.0+-blue.svg)](https://www.pytorchlightning.ai/)

## 项目概述

CLIP-Voters是一个基于CLIP模型和Dempster-Shafer证据理论的分布外(Out-of-Distribution, OOD)检测框架。该项目旨在解决开放数据集分类任务中的分布外检测问题，通过高效的方法识别与训练数据分布不一致的样本。

## 主要特点

- **基于预训练大模型**：利用现有高效的CLIP等大模型作为基础，模拟现实中的分类问题，有效解决OOD检测问题
- **适配器微调**：通过Adapter模块对CLIP多模态大模型进行微调，提高解决下游任务的能力，同时降低对计算资源的需求
- **任务分解与证据理论**：将多分类任务分解为多个二分类任务，并结合Dempster-Shafer证据理论，在解决误分类问题的基础上实现OOD数据的有效检测
- **高效训练**：采用PyTorch-Lightning框架进行分布式训练，显著提高训练速度和资源利用效率


本项目的核心技术架构包含以下几个部分：

1. **CLIP模型及Adapter微调**：使用CLIP作为基础模型，通过Adapter模块进行轻量级微调
2. **二分类任务分解器**：将多分类问题转化为一系列二分类问题
3. **证据理论融合模块**：基于Dempster-Shafer理论融合多个分类器的结果
4. **OOD检测器**：根据融合结果识别并筛选分布外数据


训练压力很小，欢迎尝试呀━(*｀∀´*)ノ亻!

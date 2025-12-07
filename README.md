# 大模型微调作业：基于Qwen2.5-1.5B的儿科领域大模型微调

## 项目框架

```#
Fine-tuning-of-large-models/
|-- src/
|	|-- fine_tune.py # 微调主程序
│ 	|-- chat_fine.py # 模型交互脚本
│ 	|-- predata.py # 数据预处理脚本
|-- data/ # 数据集目录
|	|-- data.csv #数据集
|	|-- train.jsonl # 训练集
|	|-- test.jsonl # 测试集
|-- scripts/
|	|--run.sh # 运行脚本
|-- qwen-1.5b # 原始模型
|-- requirements.txt # 依赖库列表
|-- qwen_medical_lora/ # 微调后模型输出目录
|--	result/ # 微调结果图像
```

# 环境安装

建议使用 Python 3.8+ 和 CUDA 环境

# 微调训练

我们提供了一键运行脚本。请确保已下载 Qwen 的基础模型权重或指定 HuggingFace 模型路径。

```bash
bash scripts/run.sh
```

训练完成后，LoRA 权重将保存在 `qwen_medical_lora/` 目录下，各项指标在`results/` 下。

## 说明

微调产出的权重文件 'adapter_model.safetensors' 和原始模型 /qwen-1.5b 未包含在本仓库中，如需获取该文件，请联系: `25125334@bjtu.edu.cn`。
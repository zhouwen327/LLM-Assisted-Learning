# AI 文旅大模型 + 漫游系统设计



## 下载基础的QWen2.5B 大模型

- [QWen2.5](https://www.modelscope.cn/models/qwen/Qwen2.5-0.5B/summary)

```python
!pip install modelscope
!modelscope download --model "Qwen/Qwen2.5-0.5B" --local_dir ./QWen2.5
```

## 安装lora 等第三方包
- requirements 

```bash
pip install torch
pip install transformers
pip install peft
pip install datasets
pip install bitsandbytes
pip install accelerate
pip install elasticsearch
pip install fastapi
pip install uvicorn
pip install flask flask_cors
```

## 准备训练语料数据集

- tourism_data.json
```json
[
    {
        {
            "instruction": "介绍中江桥",
            "input": "",
            "output": "中江桥是芜湖市跨越长江的重要桥梁，位于滨江公园东端，建成于1997年。全长约2.3公里，是连接弋江区和镜湖区的重要交通枢纽。桥梁设计新颖，两端各有一座独特的塔楼，夜间灯光璀璨，是芜湖市的标志性建筑之一。"
        }
        #......
    }
]
```

## 训练大模型 (可选)
```bash
python train_lora.py

```
## 增加知识图谱取代大模型训练，针对样本数量不足的情况

```
knowledges.py
```

## 启动HTTPS服务 (可选)

- 生成对localhost的自签名证书

```bash
openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes -subj "/CN=localhost"
```

## 运行程序

- 启动程序

```bash
python api_service.py --port 8000
```

- 客户端访问
```
http://(IP address):8000/static/index.html 
```

或 [http://127.0.0.1:8000/static/index.html](http://127.0.0.1:8000/static/index.html)


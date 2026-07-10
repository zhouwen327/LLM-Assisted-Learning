import torch
import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# ========== 本地模型路径配置 ==========
LOCAL_MODEL_PATH = "./QWen2.5"  
LORA_OUTPUT_DIR = "./qwen_lora_weights"

# ========== 本地模型存在 ==========
config_path = os.path.join(LOCAL_MODEL_PATH, "config.json")
if not os.path.exists(config_path):
    print(f"错误：找不到本地模型配置文件")
    print(f"  期望位置：{config_path}")
    print(f"\n请先执行以下步骤：")
    print(f"1. 创建目录：mkdir -p {LOCAL_MODEL_PATH}")
    print(f"2. 执行：python download_model.py")
    exit(1)

print(f"✓ 检测到本地模型: {LOCAL_MODEL_PATH}\n")

# ========== 加载本地模型与分词器（离线模式） ==========
print("正在加载分词器（离线模式）...")
tokenizer = AutoTokenizer.from_pretrained(
    LOCAL_MODEL_PATH,
    local_files_only=True,     
    trust_remote_code=True
)
print("分词器加载完成")

print("正在加载模型（FP16 + 离线模式）...")
model = AutoModelForCausalLM.from_pretrained(
    LOCAL_MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="auto",
    local_files_only=True,     # 强制本地离线模式
    trust_remote_code=True
)
print(f"模型加载完成，参数量: {model.num_parameters()/1e9:.2f}B\n")

# ========== LoRA配置 ==========
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=4,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ========== 准备数据集 ==========
qa_data_file = "./data/tourism_qa.json"

if os.path.exists(qa_data_file):
    print(f"\n从文件加载QA数据: {qa_data_file}")
    with open(qa_data_file, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
else:
    print("\n使用内联数据...")
    qa_data = [
        {
            "instruction": "介绍中江桥",
            "input": "",
            "output": "中江桥是芜湖市跨越长江的重要桥梁，位于滨江公园东端，建成于1997年。全长约2.3公里，是连接弋江区和镜湖区的重要交通枢纽。桥梁设计新颖，夜间灯光璀璨，是芜湖市的标志性建筑。"
        },
        {
            "instruction": "芜湖滨江公园有哪些景点",
            "input": "",
            "output": "滨江公园沿长江河滨带分布，主要景点包括书屋、滨江广场、滨江雕塑园、长江大堤步道。公园全长12公里，是市民休闲散步的热门地点。"
        },
        {
            "instruction": "怎么从书屋走到中江桥",
            "input": "",
            "output": "从书屋出发，沿滨江公园步道向东行走约10公里，即可到达中江桥。步道沿长江河畔，风景优美。步行时间约2-3小时。"
        },
        {
            "instruction": "滨江公园最佳游览时间",
            "input": "",
            "output": "滨江公园四季皆宜，春季百花盛开，秋季凉爽舒适最佳。"
        },
        {
            "instruction": "芜湖特色美食有哪些",
            "input": "",
            "output": "芜湖特色美食包括三鲜鱼汤、芜湖臭干子、虾子面等，其中三鲜鱼汤采用长江鱼制作。"
        },
    ]
    # 注：实际使用需补充至200条

print(f"加载了{len(qa_data)}条QA数据")

def format_qa(example):
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output_text = example.get('output', '')
    
    if input_text:
        prompt = f"用户问：{instruction}\n具体信息：{input_text}\n助手答："
    else:
        prompt = f"用户问：{instruction}\n助手答："
    
    return {"text": prompt + output_text}

dataset = Dataset.from_dict({
    "instruction": [d.get("instruction", "") for d in qa_data],
    "input": [d.get("input", "") for d in qa_data],
    "output": [d.get("output", "") for d in qa_data]
})

dataset = dataset.map(format_qa, remove_columns=["instruction", "input", "output"])

print("正在对数据进行分词...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=32,
    remove_columns=["text"]
)

# ========== 训练参数 ==========
training_args = TrainingArguments(
    output_dir=LORA_OUTPUT_DIR,
    learning_rate=1e-4,
    lr_scheduler_type="linear",
    warmup_steps=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=10,
    fp16=True,
    optim="adamw_torch",
    weight_decay=0.01,
    logging_steps=10,
    save_steps=50,
    save_total_limit=2,
    logging_first_step=True,
    seed=42,
    dataloader_pin_memory=True,
    remove_unused_columns=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ========== 训练 ==========
print("\n" + "="*60)
print("开始LoRA微调训练...")
print("="*60 + "\n")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

# ========== 保存 ==========
print("\n保存LoRA权重...")
model.save_pretrained(LORA_OUTPUT_DIR)
tokenizer.save_pretrained(LORA_OUTPUT_DIR)

with open(os.path.join(LORA_OUTPUT_DIR, "training_config.json"), 'w') as f:
    json.dump({
        "model_path": LOCAL_MODEL_PATH,
        "lora_r": 4,
        "lora_alpha": 32,
        "num_qa_pairs": len(qa_data),
        "training_epochs": 1
    }, f, indent=2, ensure_ascii=False)

print(f"\nLoRA微调完成！")
print(f"权重已保存至: {LORA_OUTPUT_DIR}")
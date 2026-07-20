from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import json
import torch
import traceback
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from elasticsearch import Elasticsearch
import asyncio
from datetime import datetime

from knowledges import *

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
tokenizer = None
es_client = None

# RAG配置
KB_INDEX = "tourism_qa"
RAG_SCORE_THRESHOLD = 10.0  
RAG_TOP_K = 3


kb = None

@app.on_event("startup")
async def load_model():
    global model, tokenizer, es_client,kb
    try:
        lora_checkpoint = "./qwen_lora_weights"
        print("正在加载模型...")

        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_checkpoint,
            dtype=torch.float32,
            device_map="cpu"
        )
        model = model.merge_and_unload()
        print("模型合并完成")

        tokenizer = AutoTokenizer.from_pretrained(lora_checkpoint)
        print("Tokenizer加载完成")

        kb = KnowledgeBase("data/tourism_qa.json")

        try:
            es_client = Elasticsearch(["http://localhost:9200"])
            if es_client.ping():
                print("Elasticsearch已连接")
            else:
                es_client = None
                print("Elasticsearch无响应，将使用纯模型模式")
        except Exception as e:
            es_client = None
            print(f"Elasticsearch不可用: {e}")

        print("模型和服务已启动")
        kb = KnowledgeBase("data/tourism_qa.json")
        print("知识库已就绪")

    except Exception as e:
        print(f"启动失败: {e}")
        traceback.print_exc()


def retrieve_from_kb(query: str):
    """返回 (context_text, hit_count, best_score)"""
    if not kb:
        return None, 0, 0.0
    hits = kb.search(query, top_k=RAG_TOP_K)
    if not hits:
        return None, 0, 0.0

    best_score, best_entry = hits[0]
    print(f"检索最高分: {best_score:.2f} | 命中问题: {best_entry['question'][:30]}")

    if best_score < RAG_SCORE_THRESHOLD:
        return None, 0, best_score

    answer = best_entry['answer']
    if not answer.endswith(("。", "！", "?", "？")):
        answer += "。"
    return best_entry['answer'], 1, best_score


def inference_sync(query: str, use_rag: bool = True):
    """
    同步推理
    返回: (answer, source, debug_info)
    命中知识库 -> 直接返回原答案，跳过模型生成
    未命中     -> 走通用模型生成
    """
    if use_rag:
        kb_answer, hit_count, max_score = retrieve_from_kb(query)
        if kb_answer:
            debug = {"hit_count": hit_count, "max_score": round(max_score, 2)}
            return kb_answer, "knowledge_base", debug
    else:
        max_score = 0.0

    # 未命中知识库，走通用模型
    prompt = (
        "你是芜湖滨江公园的AI导游，请简要回答游客的问题。"
        "如果问题与芜湖文旅无关，也请礼貌作答。\n\n"
        f"问题：{query}\n回答："
    )

    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    debug = {"hit_count": 0, "max_score": round(max_score, 2)}
    return response, "general_model", debug

async def run_inference(query: str, use_rag: bool = True):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, lambda: inference_sync(query, use_rag))


@app.post("/query")
async def query_endpoint(request: dict):
    device_id = request.get("device_id", "unknown")
    query = request.get("query", "").strip()
    use_rag = request.get("use_rag", True)

    print(f"[{device_id}] 收到查询: {query}")

    if not query:
        return JSONResponse({"status": "error", "error": "查询为空"}, status_code=400)

    try:
        answer, source, debug = await run_inference(query, use_rag)
        print(f"[{device_id}] 推理完成 (来源: {source}, 命中: {debug['hit_count']})")

        return JSONResponse({
            "status": "success",
            "device_id": device_id,
            "query": query,
            "answer": answer,
            "source": source,        # knowledge_base / general_model
            "debug": debug,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        print(f"[{device_id}] 错误: {e}")
        traceback.print_exc()
        return JSONResponse({
            "status": "error",
            "error": str(e),
            "device_id": device_id
        }, status_code=500)


@app.websocket("/ws/sync")
async def websocket_sync_endpoint(websocket: WebSocket):
    await websocket.accept()
    device_id = None

    try:
        while True:
            message = await websocket.receive_text()
            data = json.loads(message)

            msg_type = data.get("type")
            device_id = data.get("device_id")
            payload = data.get("payload", {})

            print(f"[{device_id}] {msg_type}: {payload}")

            if msg_type == "handshake":
                await websocket.send_json({
                    "type": "handshake_ack",
                    "status": "connected",
                    "server_timestamp": datetime.now().isoformat()
                })

            elif msg_type == "heartbeat":
                await websocket.send_json({
                    "type": "heartbeat_ack",
                    "sync_delay_ms": 50
                })

            elif msg_type == "screen_interaction":
                hotspot_id = payload.get("hotspot_id")
                query = f"介绍{hotspot_id}"
                answer, source, debug = await run_inference(query, use_rag=True)

                await websocket.send_json({
                    "type": "screen_response",
                    "hotspot_id": hotspot_id,
                    "answer": answer,
                    "source": source,
                    "target_devices": ["screen_001"]
                })

                await websocket.send_json({
                    "type": "vr_sync_command",
                    "action": "rotate_to_hotspot",
                    "hotspot_id": hotspot_id,
                    "target_devices": ["vr_001"]
                })

            elif msg_type == "vr_speech_query":
                query_text = payload.get("query_text")
                answer, source, debug = await run_inference(query_text, use_rag=True)

                await websocket.send_json({
                    "type": "vr_response",
                    "answer": answer,
                    "source": source,
                    "should_speak": True,
                    "target_devices": ["vr_001"]
                })

                await websocket.send_json({
                    "type": "screen_sync_display",
                    "content": answer,
                    "target_devices": ["screen_001"]
                })

    except Exception as e:
        print(f"WebSocket错误: {e}")
        traceback.print_exc()
    finally:
        print(f"[{device_id}] 连接已断开")


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def read_root():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

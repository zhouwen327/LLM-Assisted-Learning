from fastapi import FastAPI, WebSocket
from fastapi.responses import JSONResponse
import json
import torch
import traceback
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
from elasticsearch import Elasticsearch
import asyncio
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os


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

@app.on_event("startup")
async def load_model():
    global model, tokenizer, es_client
    try:
        lora_checkpoint = "./qwen_lora_weights"
        print("正在加载模型...")
        
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_checkpoint,
            torch_dtype=torch.float32,
            device_map="cpu"
        )
        model = model.merge_and_unload()
        print("模型合并完成")
        
        tokenizer = AutoTokenizer.from_pretrained(lora_checkpoint)
        print("Tokenizer加载完成")
        
        try:
            es_client = Elasticsearch(["http://localhost:9200"])
            print("Elasticsearch已连接")
        except Exception as e:
            es_client = None
            print(f"Elasticsearch不可用: {e}")
        
        print("模型和服务已启动")
    
    except Exception as e:
        print(f"启动失败: {e}")
        traceback.print_exc()

@app.post("/query")
async def query_endpoint(request: dict):
    device_id = request.get("device_id", "unknown")
    query = request.get("query", "").strip()
    use_rag = request.get("use_rag", True)
    
    print(f"[{device_id}] 收到查询: {query}")
    
    if not query:
        return JSONResponse({"status": "error", "error": "查询为空"}, status_code=400)
    
    try:
        print(f"[{device_id}] 开始推理...")
        answer = await run_inference(query, use_rag)
        print(f"[{device_id}] 推理完成")
        
        return JSONResponse({
            "status": "success",
            "device_id": device_id,
            "query": query,
            "answer": answer,
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

async def run_inference(query: str, use_rag: bool = True):
    """推理函数（异步包装）"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: inference_sync(query, use_rag)
    )

def inference_sync(query: str, use_rag: bool = True):
    """同步推理（CPU上运行）"""
    try:
        if use_rag and es_client:
            try:
                results = es_client.search(
                    index="tourism_qa",
                    body={"query": {"match": {"question": query}}, "size": 3}
                )
                context_text = "\n".join([
                    hit["_source"]["answer"] for hit in results["hits"]["hits"]
                ])
                prompt = f"背景：{context_text}\n\n{query}"
            except:
                prompt = query
        else:
            prompt = query
        
        inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    
    except Exception as e:
        print(f"推理异常: {e}")
        traceback.print_exc()
        raise

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
                query = f"关于{hotspot_id}的信息"
                answer = await run_inference(query, use_rag=True)
                
                await websocket.send_json({
                    "type": "screen_response",
                    "hotspot_id": hotspot_id,
                    "answer": answer,
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
                answer = await run_inference(query_text, use_rag=True)
                
                await websocket.send_json({
                    "type": "vr_response",
                    "answer": answer,
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
    """提供前端HTML"""
    return FileResponse("static/index.html")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
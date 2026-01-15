import os
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from .util import load_multiple_qa_files, prepareDocuments

# 数据与模型路径
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "..", "rag_data")
EMB_PATH = os.path.join(DATA_DIR, "embeddings.npy")
DOCS_PATH = os.path.join(DATA_DIR, "docs.pkl")

# 默认模型路径（可修改）
DEFAULT_MODEL_PATH = "/scripts/ORFS-Research/flow-Agent/models/mxbai-embed-large-v1"

# ============================================================
# 🔹 构建并保存 Embedding 向量库
# ============================================================
def build_and_save_embeddings(base_dir="/scripts/ORFS-Research/flow-Agent/EDA-Corpus-main/Augmented_Data/Question-Answer",
                              model_name=DEFAULT_MODEL_PATH):
    """
    从 Flow / General / Tools 三个 CSV 构建向量库并保存。
    """
    print("[RAG] 正在加载 QA 文件...")
    df = load_multiple_qa_files(base_dir)

    # 准备文档
    docs, docsDict = prepareDocuments(df)

    # 构建模型
    print(f"[RAG] 正在加载嵌入模型：{model_name}")
    model = SentenceTransformer(model_name)

    # 生成向量
    print("[RAG] 正在生成文本向量...")
    embeddings = model.encode(docs, convert_to_numpy=True, show_progress_bar=True)

    # 确保保存路径存在
    os.makedirs(DATA_DIR, exist_ok=True)

    # 保存
    np.save(EMB_PATH, embeddings)
    with open(DOCS_PATH, "wb") as f:
        pickle.dump((docs, docsDict), f)

    print(f"[RAG] ✅ 向量库构建完成并保存到 {DATA_DIR}")


# ============================================================
# 🔹 加载已保存的向量库
# ============================================================
def load_embeddings_and_docs():
    if not os.path.exists(EMB_PATH) or not os.path.exists(DOCS_PATH):
        raise FileNotFoundError("[RAG] 向量库文件不存在，请先运行 build_and_save_embeddings()")

    embeddings = np.load(EMB_PATH)
    with open(DOCS_PATH, "rb") as f:
        docs, docsDict = pickle.load(f)

    print(f"[RAG] 成功加载向量库，共 {len(docs)} 条文档。")
    return embeddings, docs, docsDict

# ===========================
# 加载索引并执行相似性搜索
# ===========================
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 配置全局嵌入模型
Settings.embed_model = HuggingFaceEmbedding("BAAI/bge-small-zh-v1.5")

# 从本地持久化目录加载 storage_context
persist_path = "./llamaindex_index_store"
storage_context = StorageContext.from_defaults(persist_dir=persist_path)

# 加载索引（会使用 Settings.embed_model 进行向量化/检索）
loaded_index = load_index_from_storage(storage_context)

# 用 retriever 做相似性检索
query = "LlamaIndex是做什么的？"
retriever = loaded_index.as_retriever(similarity_top_k=1)
nodes = retriever.retrieve(query)

print(f"\n查询: '{query}'")
print("相似度最高的文档:")
for node in nodes:
    # node.node.text 是原文内容；node.score 是相似度分数（越大越相关，具体尺度取决于向量库实现）
    print(f"- {node.node.text}")
    print(f"  score: {node.score}")
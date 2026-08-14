"""L3.5 SHAP+LLM 临床解释层。

模块构成：
    - config: 解释层配置加载
    - feature_meta: 特征元数据白名单
    - knowledge_builder: 知识库文档切分与索引构建
    - rag_retriever: FAISS 向量检索（元数据过滤 + 路由）
    - llm_client: Agnes AI（OpenAI 兼容）调用封装
    - prompts: System/User Prompt 与 Output Schema
    - shap_structurer: SHAP 原始值 → 结构化 JSON
    - reference_validator: 关键事实核对（数值匹配）
    - shap_llm: 主入口 generate_explanation()

契约：严格遵循 L1→L2→L3→L3.5→L4→L5 分层，L3.5 不反向调用 L3。
"""

from domain.explain.shap_llm import generate_explanation

__all__ = ["generate_explanation"]

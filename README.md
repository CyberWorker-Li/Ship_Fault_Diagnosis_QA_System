   Ship Fault AI Assistant 项目介绍
==========================================================================

1. 项目简介
-----------
本项目是一个针对课程文档（.docx/.pdf/.txt）构建的智能问答助手，核心目标是：
- 证据可追溯：回答要求给出引用编号
- 复杂问题可回答：Graph-RAG（图谱证据）+ 混合检索（向量 + 关键词）
- 本地优先：默认使用本地 Ollama，本地 embedding/rerank；可选接入 DeepSeek 做润色与图谱清洗

2. 技术路线 (Technical Stack)
----------------------------
* 核心框架: LangChain（流程编排、文本切分、LLM 调用）
* 本地语言模型(LLM): DeepSeek-R1 7B via Ollama（当前默认 `deepseek-r1:7b`）
* 文本向量化(Embedding): BAAI/bge-m3（本地加载，多语言/长文本）
* 重排模型(Reranker): BAAI/bge-reranker-base（本地 cross-encoder 精排）
* 关键词检索: Rank-BM25 + Jieba
* 向量索引: FAISS（本地）
* 知识图谱:
  - NetworkX（进程内图，用于兼容与可视化）
  - Neo4j（可选后端，用于持久化与更快的子图查询）

3. 支持的文档格式与数据目录
--------------------------
数据目录默认: AI_Assistant/data
支持读取:
- .docx
- .pdf
- .txt
不支持直接读取:
- .doc（旧 Word 格式），需先转换为 .docx
备注:
- Office 可能生成 ~$.docx 临时文件，建议从 data 目录移除/忽略

4. 运行逻辑与过程 (Core Workflow)
--------------------------------
A. 启动阶段 (Initialization)
1) 文档加载: 从 data 目录读取文档
2) 智能切分: RecursiveCharacterTextSplitter（chunk_size/chunk_overlap 可配置）
3) 向量构建: bge-m3 生成 embedding，构建 FAISS
4) 关键词构建: jieba 分词，构建 BM25
5) 知识图谱（可选）:
   - enable_graph_on_start=true 时执行三元组抽取并建图
   - 若启用 Neo4j，则将三元组写入 Neo4j（:Entity)-[:REL {label}]->(:Entity)
   - 支持从本地 graph_store.pkl 加载/保存缓存（加速后续启动）

B. 问答阶段 (Query Process)
1) NLU: 识别意图/实体/复杂度（simple vs multi-hop）
2) 查询改写（可选）: LLM rewrite 生成更适合检索的 query
3) 混合检索: 向量检索 + BM25
4) 融合: RRF 合并排序
5) 图谱增强（可选）: 基于实体从图中提取邻居/路径作为“图证据”
6) 精排（可选）: bge-reranker-base 精排候选证据
7) 证据压缩与上下文扩展: 选取片段并抓取相邻块
8) 生成回答: LLM 基于证据生成回答，并输出引用编号
9) 润色（可选）: 通过 DeepSeek 对最终文本润色（保留引用编号）
10) 小范围上下文记忆（可选）: 将最近几轮问答作为“理解指代”的辅助上下文（与证据冲突时以证据为准）

5. 快速开始 (Windows)
--------------------
1) 安装依赖（示例）
   pip install langchain langchain-openai langchain-text-splitters
   pip install sentence-transformers faiss-cpu rank-bm25 jieba networkx requests neo4j
2) 启动本地 Ollama，并拉取模型
   ollama pull deepseek-r1:7b
3) 模型准备（重要）
   - 本项目 `models/` 目录体积较大，通常不提交到代码仓库/网盘。
   - 请在本机准备以下目录（可从原项目机器拷贝）：
     - `models/bge-m3`
     - `models/bge-reranker-base`
   - 若未拷贝，可按需从 HuggingFace 下载同名模型到本地目录。
4) 将课程文档放入 AI_Assistant/data（.docx/.pdf/.txt）
5) 启动
   双击 AI_Assistant/run.bat

6. run.bat 启动脚本说明
----------------------
run.bat 作用:
- 自动定位 AI_Assistant 目录并设置 PYTHONPATH
- 检查 Python 与 Ollama 服务是否可用
- 通过环境变量注入所有运行时开关（LLM/检索/图谱/Neo4j/润色等）
- 当前固定 UI 启动模式（CLI/ALL 入口已禁用）:
  - UI: python -m streamlit run ui\app.py
  - 传入 `cli` 或 `all` 参数时会自动提示并回退到 UI 模式

用法:
- 双击 run.bat：直接启动 UI
- 命令行参数：可传 `cli`/`all`，但会被脚本提示并自动切换为 UI

UI 说明（Streamlit）:
- 问答：聊天式对话（支持多轮追问），可按需展开证据/Prompt/调试信息
- 图谱可视化：按实体关键词、跳数、节点/边上限查看子图
- 专家审图：候选边列表 + 搜索（按节点/按关系）+ Approve/Reject/Accept Suggestion/Set Label
- AI助审：在“专家审图”面板点击“AI助审”触发 `tools/deepseek_graph_audit.py`，自动生成关系复核建议
- 增量导入：在侧边栏“资料增量导入”上传 `.docx/.pdf/.txt`，系统先写入 `data/new`，处理后自动并入知识库并迁移到 `data`

你常用会改的变量（都在 run.bat 的 set 行里）:
- LLM（Ollama）:
  AI_ASSISTANT_LLM_BASE_URL
  AI_ASSISTANT_LLM_ANSWER_MODEL
  AI_ASSISTANT_LLM_RERANK_MODEL
- DeepSeek 润色:
  AI_ASSISTANT_ENABLE_LLM_POLISH
  AI_ASSISTANT_POLISH_BASE_URL
  AI_ASSISTANT_POLISH_MODEL
  AI_ASSISTANT_POLISH_API_KEY
- Neo4j:
  AI_ASSISTANT_ENABLE_NEO4J_GRAPH
  AI_ASSISTANT_NEO4J_URI / AI_ASSISTANT_NEO4J_USER / AI_ASSISTANT_NEO4J_PASSWORD / AI_ASSISTANT_NEO4J_DATABASE
  AI_ASSISTANT_NEO4J_CLEAR_ON_BUILD
- 启动建图（建议日常关闭）:
  AI_ASSISTANT_ENABLE_GRAPH_ON_START
- 输出与会话:
  AI_ASSISTANT_ENABLE_DEBUG_OUTPUT
  AI_ASSISTANT_CONCISE_ANSWER
  AI_ASSISTANT_ENABLE_SESSION_MEMORY / AI_ASSISTANT_SESSION_MAX_TURNS / AI_ASSISTANT_SESSION_MAX_CHARS

安全提醒:
- 不建议把 DeepSeek API Key 明文提交到仓库；如团队协作必须写入 run.bat，请使用可随时轮换的低权限 key。
- 不建议提交 `models/`、`data/`、`.cache/` 等大文件目录，建议由部署文档指引本地准备。

7. 关键配置说明 (Configuration)
------------------------------
配置来源:
- 代码默认值: knowledge/shared/config.py
- 运行时环境变量: run.bat 中 set 的变量

常用参数:
- 检索与切分:
  AI_ASSISTANT_TOP_K（默认3）
  AI_ASSISTANT_RETRIEVAL_CANDIDATE_K（默认8，融合/精排前的候选规模）
  AI_ASSISTANT_DOC_TOP_N（默认2，跨文档的候选上限，用于压制单文档刷屏）
  AI_ASSISTANT_CHUNK_SIZE（默认800）
  AI_ASSISTANT_CHUNK_OVERLAP（默认200）
- LLM(Ollama):
  AI_ASSISTANT_LLM_BASE_URL（默认 http://localhost:11434/v1）
  AI_ASSISTANT_LLM_ANSWER_MODEL（当前默认 deepseek-r1:7b）
  AI_ASSISTANT_ENABLE_LLM_REWRITE / AI_ASSISTANT_ENABLE_LLM_RERANK
- DeepSeek 润色:
  AI_ASSISTANT_ENABLE_LLM_POLISH
  AI_ASSISTANT_POLISH_BASE_URL=https://api.deepseek.com/v1
  AI_ASSISTANT_POLISH_MODEL=deepseek-chat
  AI_ASSISTANT_POLISH_API_KEY
- 启动建图/缓存:
  AI_ASSISTANT_ENABLE_GRAPH_ON_START（建议日常为 false；仅在数据更新后手动打开一次）
  AI_ASSISTANT_GRAPH_CACHE_PATH（默认 data/graph_store.pkl）
  AI_ASSISTANT_GRAPH_BUILD_MAX_CHUNKS（限制建图块数，降低资源压力）
  AI_ASSISTANT_ENABLE_INDEX_CACHE（默认 true，缓存 chunks/BM25/向量索引以加速启动）
  AI_ASSISTANT_CACHE_DIR（默认 data/.cache）
  AI_ASSISTANT_CACHE_VERSION（默认 v1，变更缓存格式/策略时可 bump 以自动失效旧缓存）
- Neo4j:
  AI_ASSISTANT_ENABLE_NEO4J_GRAPH
  AI_ASSISTANT_NEO4J_URI（默认 bolt://localhost:7687）
  AI_ASSISTANT_NEO4J_USER / AI_ASSISTANT_NEO4J_PASSWORD / AI_ASSISTANT_NEO4J_DATABASE
  AI_ASSISTANT_NEO4J_CLEAR_ON_BUILD（重建时是否清库）
- 图输出（可选）:
  AI_ASSISTANT_GRAPH_HTML（默认 knowledge_graph.html）
- 小范围会话记忆:
  AI_ASSISTANT_ENABLE_SESSION_MEMORY（默认 true）
  AI_ASSISTANT_SESSION_MAX_TURNS（默认4）
  AI_ASSISTANT_SESSION_MAX_CHARS（默认1200）
- 输出模式:
  AI_ASSISTANT_ENABLE_DEBUG_OUTPUT（调试输出）
  AI_ASSISTANT_CONCISE_ANSWER（简洁输出，仅最终答案）

8. Neo4j 使用说明
----------------
1) Neo4j Desktop 创建本地实例并启动（Bolt 通常为 7687）
2) 在 run.bat 设置连接参数（uri/user/password/database）
3) 仅在数据更新时启用建图导入:
   set AI_ASSISTANT_ENABLE_GRAPH_ON_START=true
4) 导入完成后建议改回 false，加速日常启动
5) 验证（Neo4j Query 工具）
   MATCH (n:Entity) RETURN count(n);
   MATCH (:Entity)-[r:REL]->(:Entity) RETURN count(r);

9. 图谱清洗（DeepSeek 复核脚本）
------------------------------
脚本位置: AI_Assistant/tools/deepseek_graph_audit.py
用途:
- 批量抽检/复核图谱关系，写回 deepseek_verdict/confidence/reason/suggested_label 等字段
- 支持对明显噪声做规则过滤；对“关系名不贴切”给出 suggested_label，供专家一键采纳

成本与安全阀（重要）:
- 默认 dry-run：不写库；只有加 --apply 才会写回 Neo4j
- 可设置预算与 token 上限：--budget-rmb / --max-total-tokens / --max-output-tokens
- 可控制抽检规模：--max-edges / --batch-size / --max-requests
- 可选择“只标注不拒绝”：把 auto-reject-threshold 设得很高，避免脚本自动把边置为 rejected

说明:
- 脚本是独立工具，不影响主问答流程；即使脚本异常退出，也不会影响 CLI/UI 的问答。

10. 常见问题
------------
- jieba 报 pkg_resources deprecate 警告：不影响运行，可忽略。
- .doc 无法读取：先转换为 .docx。
- 启动很慢：可能在建图/抽三元组；日常将 AI_ASSISTANT_ENABLE_GRAPH_ON_START 设为 false。
- Ollama 500 runner stopped：资源不足导致，降低建图块数或换更轻量模型进行抽取。
- Docker build 过程中出现 failed to receive status / EOF：常见于 Docker Desktop 引擎崩溃或网络抖动，建议重启 Docker Desktop、执行 wsl --shutdown 后重试。

11. 大模型目录过大时的配置与指导
-------------------------------
1) 为什么不上传 `models/`
- `models/` 体积通常较大（数 GB~数十 GB），会导致代码托管平台、网盘或邮件传输失败。
- 推荐做法：代码与配置单独上传，模型目录通过“离线拷贝”或“首次部署下载”方式准备。

2) 非 Docker 环境（run.bat）如何配置
- 在 `run.bat` 保持或修改为本地路径：
  - `set AI_ASSISTANT_EMBEDDING_MODEL=%PROJECT_ROOT%models\bge-m3`
  - `set AI_ASSISTANT_RERANKER_MODEL=%PROJECT_ROOT%models\bge-reranker-base`
- 若模型放在其他盘符（如 `D:\models`），可改为绝对路径。

3) Docker 环境如何配置（若使用）
- `docker-compose.yml` 通过卷挂载本地模型目录：
  - `./models:/app/models`
- 容器内环境变量应指向：
  - `AI_ASSISTANT_EMBEDDING_MODEL=/app/models/bge-m3`
  - `AI_ASSISTANT_RERANKER_MODEL=/app/models/bge-reranker-base`

4) 新机器迁移最小步骤
- 步骤1：先拉取代码（不含 models/data）
- 步骤2：拷贝 `models/bge-m3` 与 `models/bge-reranker-base`
- 步骤3：准备 `data/` 文档
- 步骤4：运行 `run.bat` 验证启动

12. 日常运维建议（非 Docker）
--------------------------
1) 日常模式（推荐）:
- set AI_ASSISTANT_ENABLE_GRAPH_ON_START=false
- set AI_ASSISTANT_NEO4J_CLEAR_ON_BUILD=false

2) 数据更新后重建（一次性）:
- set AI_ASSISTANT_ENABLE_GRAPH_ON_START=true
- set AI_ASSISTANT_NEO4J_CLEAR_ON_BUILD=true
- 重建完成后务必改回“日常模式”。

3) AI助审配置优先级:
- 优先读取 AI_ASSISTANT_GRAPH_AUDIT_BASE_URL / MODEL / API_KEY
- 未配置时回退到 POLISH 或 LLM 配置。



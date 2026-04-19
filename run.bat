@echo off
SETLOCAL

:: 1. 设置编码为UTF-8，防止中文乱码
chcp 65001 > nul

echo ======================================================
echo    Ship Fault Diagnosis QA Assistant 启动程序
echo ======================================================

:: 2. 自动定位项目根目录
set PROJECT_ROOT=%~dp0
cd /d %PROJECT_ROOT%

:: 设置PYTHONPATH，确保模块导入不会报错 (解决 ModuleNotFoundError)
set PYTHONPATH=%PROJECT_ROOT%

:: 3. LLM 核心配置 (本地 Ollama 模式)
:: 如果未来更换模型或端口，直接修改这里即可
set AI_ASSISTANT_LLM_FRAMEWORK=langchain
set AI_ASSISTANT_LLM_BASE_URL=http://localhost:11434/v1
set AI_ASSISTANT_LLM_API_KEY=no-key
set AI_ASSISTANT_LLM_TIMEOUT=180
set AI_ASSISTANT_LLM_ANSWER_MODEL=deepseek-r1:7b
set AI_ASSISTANT_LLM_RERANK_MODEL=deepseek-r1:7b
set AI_ASSISTANT_EMBEDDING_MODEL=%PROJECT_ROOT%models\bge-m3
set AI_ASSISTANT_RERANKER_MODEL=%PROJECT_ROOT%models\bge-reranker-base

:: 4. 功能增强配置
set AI_ASSISTANT_ENABLE_LLM_REWRITE=true
set AI_ASSISTANT_ENABLE_LLM_RERANK=true
set AI_ASSISTANT_ENABLE_RERANK=true
set AI_ASSISTANT_LOCAL_FILES_ONLY=true
set AI_ASSISTANT_ENABLE_LLM_POLISH=false
set AI_ASSISTANT_POLISH_BASE_URL=
set AI_ASSISTANT_POLISH_MODEL=deepseek-chat
set AI_ASSISTANT_POLISH_API_KEY=

set AI_ASSISTANT_ENABLE_NEO4J_GRAPH=true
set AI_ASSISTANT_NEO4J_URI=bolt://localhost:7687
set AI_ASSISTANT_NEO4J_USER=neo4j
set AI_ASSISTANT_NEO4J_PASSWORD=123456789
set AI_ASSISTANT_NEO4J_DATABASE=ship.fault.qa
set AI_ASSISTANT_NEO4J_CLEAR_ON_BUILD=false

set AI_ASSISTANT_ENABLE_GRAPH_ON_START=false

:: 5. 环境验证
echo [*] 正在验证运行环境...

:: 检查 Python
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo [错误] 未找到 Python，请确保已安装并加入系统环境变量。
    pause
    exit /b 1
)

:: 检查 Ollama 服务状态
curl -s http://localhost:11434/api/tags > nul
if %ERRORLEVEL% NEQ 0 (
    echo [警告] 无法连接到 Ollama 服务！
    echo 请确保 Ollama 软件已启动（查看任务栏图标）。
    echo ------------------------------------------------------
    pause
) else (
    echo [OK] Ollama 服务在线，本地模型就绪。
)

:: 6. 启动程序（仅UI）
set MODE=%1
if /I "%MODE%"=="cli" (
    echo [提示] CLI 启动入口已禁用，已自动切换为 UI 模式。
)
if /I "%MODE%"=="all" (
    echo [提示] ALL 启动入口已禁用，已自动切换为 UI 模式。
)

echo [*] 正在启动 Web UI...
echo ------------------------------------------------------
python -m streamlit run ui\app.py
goto RUN_END

:RUN_END
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [提示] 程序异常退出，请检查上方报错信息。
    pause
)

ENDLOCAL
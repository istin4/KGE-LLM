# KGE-LLM
A framework for enhancing LLM
使用知识图谱增强大模型的框架

# 部署与运行
1. 安装requirements.txt中的必要依赖
   ```
   pip install -r requirements.txt
   ```
3. 安装并启动neo4j
   ```
   cd neo4j-your_version
   bin/neo4j start
   ```
5. 命令提示行中输入
   ```
   python src/scripts/api_server_v3/openai_api_server.py --base_model path/to/your/model --gpus 0 --use_flash_attention_2
   ```

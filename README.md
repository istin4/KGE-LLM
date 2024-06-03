# KGE-LLM
A framework for enhancing LLM
使用知识图谱增强大模型的框架

# 部署与运行
1. 安装requirements.txt中的必要依赖
   ```
   pip install -r requirements.txt
   ```
2. 下载LLaMA-3模型，推荐使用
   ```
   git clone https://huggingface.co/hfl/llama-3-chinese-8b-instruct-v3
   ```
3. 安装并启动neo4j
   ```
   cd neo4j-version
   bin/neo4j start
   ```
   使用如下命令导入清洗后的知识图谱
   ```
   bin/neo4j-admin database import full --nodes=' ../kg/vertex_output_vertex_all.csv' --relationships='../kg/edge_output_all.csv' --skip-duplicate-nodes=true --skip-bad-relationships=true --id-type=string neo4j
   ```
4. 命令提示行中输入如下命令以启动server
   ```
   python src/scripts/api_server_v3/openai_api_server.py --base_model path/to/your/model --gpus 0 --use_flash_attention_2
   ```

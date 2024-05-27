import argparse
import os
import time
import operator
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from threading import Thread
from sse_starlette.sse import EventSourceResponse

from jsonformer import Jsonformer

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', default=None, type=str, required=True)
parser.add_argument('--lora_model', default=None, type=str,help="If None, perform inference on the base model")
parser.add_argument('--tokenizer_path',default=None,type=str)
parser.add_argument('--gpus', default="0", type=str)
parser.add_argument('--load_in_8bit',action='store_true', help='Load the model in 8bit mode')
parser.add_argument('--load_in_4bit',action='store_true', help='Load the model in 4bit mode')
parser.add_argument('--only_cpu',action='store_true',help='Only use CPU for inference')
parser.add_argument('--alpha',type=str,default="1.0", help="The scaling factor of NTK method, can be a float or 'auto'. ")
parser.add_argument('--use_ntk', action='store_true', help="Use dynamic-ntk to extend context window")
parser.add_argument('--use_flash_attention_2', action='store_true', help="Use flash-attention2 to accelerate inference")
args = parser.parse_args()
if args.only_cpu is True:
    args.gpus = ""
    if args.load_in_8bit or args.load_in_4bit:
        raise ValueError("Quantization is unavailable on CPU.")
if args.load_in_8bit and args.load_in_4bit:
    raise ValueError("Only one quantization method can be chosen for inference. Please check your arguments")
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    LlamaTokenizer,
    GenerationConfig,
    TextIteratorStreamer,
    BitsAndBytesConfig
)
from peft import PeftModel

import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from attn_and_long_ctx_patches import apply_attention_patch, apply_ntk_scaling_patch

apply_attention_patch(use_memory_efficient_attention=True)
if args.use_ntk:
    apply_ntk_scaling_patch(args.alpha)

from openai_api_protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatCompletionResponseChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    EmbeddingsRequest,
    EmbeddingsResponse,
    ChatCompletionResponseStreamChoice,
    DeltaMessage,
)

load_type = torch.float16
if torch.cuda.is_available():
    device = torch.device(0)
else:
    device = torch.device("cpu")
if args.tokenizer_path is None:
    args.tokenizer_path = args.lora_model
    if args.lora_model is None:
        args.tokenizer_path = args.base_model
tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path, legacy=True)
if args.load_in_4bit or args.load_in_8bit:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        bnb_4bit_compute_dtype=load_type,
    )
base_model = AutoModelForCausalLM.from_pretrained(
    args.base_model,
    torch_dtype=load_type,
    low_cpu_mem_usage=True,
    device_map='auto' if not args.only_cpu else None,
    load_in_4bit=args.load_in_4bit,
    load_in_8bit=args.load_in_8bit,
    quantization_config=quantization_config if (args.load_in_4bit or args.load_in_8bit) else None,
    use_flash_attention_2=args.use_flash_attention_2,
    trust_remote_code=True
)

model_vocab_size = base_model.get_input_embeddings().weight.size(0)
tokenizer_vocab_size = len(tokenizer)
print(f"Vocab of the base model: {model_vocab_size}")
print(f"Vocab of the tokenizer: {tokenizer_vocab_size}")
if model_vocab_size != tokenizer_vocab_size:
    print("Resize model embeddings to fit tokenizer")
    base_model.resize_token_embeddings(tokenizer_vocab_size)
if args.lora_model is not None:
    print("loading peft model")
    model = PeftModel.from_pretrained(
        base_model,
        args.lora_model,
        torch_dtype=load_type,
        device_map="auto",
    )
else:
    model = base_model

if device == torch.device("cpu"):
    model.float()

model.eval()

DEFAULT_SYSTEM_PROMPT = ("""You are a helpful assistant. 你是一个乐于助人的助手。""")

CLASSIFIER_SYSTEM_PROMPT = (
    """You are a classifier that classifies user's input into three different catagories: 'chat', 'static_question' and 'dynamic_question'. To be specific, if the content contains some questions that may need to search for websites of specialized facilities to answer, such as weather or news recently, it is classified as 'dynamic_question'; if the content input by the user contains some common sense questions, and the answers are deem to remain stable for a long period of time, then it is classified as 'static_question'; In all other cases, it is classified as 'chat'. Furthermore, when you have classified the content as 'static_question', you must extract and output the keyword of the content. User's input is {input}"""
)

KEYWORD_SYSTEM_PROMPT = (
    """你是一个关键词提取器，对于用户的问题，你会分析并输出一个和对话主题关系最为密切的中文词语。用户的输入是：{input}"""
)

STATIC_SYSTEM_PROMPT = ("""You are a helpful assistant.你是一个乐于助人的助手。 注意：用户会提供给你一些三元组，它们是与问题相关的事实信息，三元组如下:{input}。这些信息是绝对可靠的，请在回答时结合给定的事实以及自身逻辑进行合理且充分的回答。""")

DYNAMIC_SYSTEM_PROMPT = ("""You are a content supervisor which only provides reliable and concrete contents. 你是一个只提供可靠信息的内容监察者。在监察的过程中，你发现用户的问题是具有极强的时效性的，只有通过搜索互联网内容才能得到正确答案，但你与互联网的连接被切断了，所以你无法对用户的问题提供可靠的答案。在这种情况下，无论你是否知道问题的答案，你只需要在回答时对用户表示歉意，并向用户解释无法回答的原因即可。""")

TEMPLATE_WITH_SYSTEM_PROMPT = (
    "[INST] <<SYS>>\n" "{system_prompt}\n" "<</SYS>>\n\n" "{instruction} [/INST]"
)

TEMPLATE_WITHOUT_SYSTEM_PROMPT = "[INST] {instruction} [/INST]"

json_schema_clsf = {
    "type": "object",
    "properties": {
        "questiontype": {"type": "string"},
        "keyword": {"type": "string"}
    },
    "required": ["questiontype"]
}

json_schema_kw = {
    "type": "object",
    "properties": {
        "keyword": {"type": "string"}
    }
}

from neo4jgraphstore import Neo4jGraphStore

username = "neo4j"
password = "dzq20020129"
url = "bolt://localhost:7687"
database = "neo4j"

# 问题初步分类
def classifier(input:str):
    prompt = CLASSIFIER_SYSTEM_PROMPT.format_map({"input":input})
    jsonformer = Jsonformer(model, tokenizer, json_schema_clsf, prompt)
    generated_data = jsonformer()
    questiontype = generated_data['questiontype']
    return questiontype

# 针对question 获取keyword
def get_entity(chat_history:str):
    # 对话历史要想办法处理一下
    input = chat_history
    prompt_kw = KEYWORD_SYSTEM_PROMPT.format_map({"input":input})
    jsonformer = Jsonformer(model, tokenizer, json_schema_kw, prompt_kw)
    generated_data = jsonformer()
    entity = generated_data['keyword']
    return entity

def retrieve_tuple(entity: str):
    graph_store = Neo4jGraphStore(
        username=username,
        password=password,
        url=url,
        database=database,
    )
    tuple = graph_store.retrieve(entity)
    return tuple



def generate_prompt(
    instruction, response="", with_system_prompt=True, system_prompt=None
):
    if with_system_prompt is True:
        if system_prompt is None:
            system_prompt = DEFAULT_SYSTEM_PROMPT
        prompt = TEMPLATE_WITH_SYSTEM_PROMPT.format_map(
            {"instruction": instruction, "system_prompt": system_prompt}
        )
    else:
        prompt = TEMPLATE_WITHOUT_SYSTEM_PROMPT.format_map({"instruction": instruction})
    if len(response) > 0:
        prompt += " " + response
    return prompt


def generate_completion_prompt(instruction: str):
    """Generate prompt for completion"""
    return generate_prompt(instruction, response="", with_system_prompt=True)


def generate_chat_prompt(messages: list):
    """Generate prompt for chat completion"""

    system_msg = None
    for msg in messages:
        if msg.role == "system":
            system_msg = msg.content
    prompt = ""
    is_first_user_content = True
    for msg in messages:
        if msg.role == "system":
            continue
        if msg.role == "user":
            if is_first_user_content is True:
                prompt += generate_prompt(
                    msg.content, with_system_prompt=True, system_prompt=system_msg
                )
                is_first_user_content = False
            else:
                prompt += "<s>" + generate_prompt(msg.content, with_system_prompt=False)
        if msg.role == "assistant":
            prompt += f" {msg.content}" + "</s>"
    return prompt

def generate_chat_prompt_with_tuple(messages: list, tuple, response=""):
    tuplestr = '\n'.join([' '.join(inner_tuple) for inner_tuple in tuple])
    print(type(tuplestr))
    system_msg = STATIC_SYSTEM_PROMPT.format_map({"input": tuplestr})
    prompt = ""
    is_first_user_content = True
    for msg in messages:
        if msg['role'] == "system":
            continue
        if msg['role'] == "user":
            if is_first_user_content is True:
                prompt += generate_prompt(
                    msg['content'], with_system_prompt=True, system_prompt=system_msg
                )
                is_first_user_content = False
            else:
                prompt += "<s>" + generate_prompt(msg['content'], with_system_prompt=False)
        if msg['role'] == "assistant":
            prompt += f" {msg['content']}" + "</s>"
    return prompt

def generate_chat_prompt_apologize(messages: list, response=""):
    system_msg = DYNAMIC_SYSTEM_PROMPT
    prompt = ""
    is_first_user_content = True
    for msg in messages:
        if msg['role'] == "system":
            continue
        if msg['role'] == "user":
            if is_first_user_content is True:
                prompt += generate_prompt(
                    msg['content'], with_system_prompt=True, system_prompt=system_msg
                )
                is_first_user_content = False
            else:
                prompt += "<s>" + generate_prompt(msg['content'], with_system_prompt=False)
        if msg['role'] == "assistant":
            prompt += f" {msg['content']}" + "</s>"
    return prompt

def predict(
    input,
    prompt=None,
    max_new_tokens=128,
    top_p=0.9,
    temperature=0.2,
    top_k=40,
    num_beams=1,
    repetition_penalty=1.1,
    do_sample=True,
    **kwargs,
):
    """
    Main inference method
    type(input) == str -> /v1/completions
    type(input) == list -> /v1/chat/completions
    """
    if prompt is None:
        if isinstance(input, str):
            prompt = generate_completion_prompt(input)
        else:
            prompt = generate_chat_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        **kwargs,
    )
    generation_config.return_dict_in_generate = True
    generation_config.output_scores = False
    generation_config.max_new_tokens = max_new_tokens
    generation_config.repetition_penalty = float(repetition_penalty)
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
        )
    s = generation_output.sequences[0]
    output = tokenizer.decode(s, skip_special_tokens=True)
    output = output.split("[/INST]")[-1].strip()
    return output

def stream_predict(
    input,
    prompt=None,
    max_new_tokens=128,
    top_p=0.75,
    temperature=0.1,
    top_k=40,
    num_beams=4,
    repetition_penalty=1.0,
    do_sample=True,
    model_id="llama2",
    **kwargs,
):
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(role="assistant"), finish_reason=None
    )
    chunk = ChatCompletionResponse(
        model=model_id,
        choices=[choice_data],
        object="chat.completion.chunk",
    )
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))

    if prompt is None:
        if isinstance(input, str):
            prompt = generate_completion_prompt(input)
        else:
            prompt = generate_chat_prompt(input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        num_beams=num_beams,
        do_sample=do_sample,
        **kwargs,
    )

    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    generation_kwargs = dict(
        streamer=streamer,
        input_ids=input_ids,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=False,
        max_new_tokens=max_new_tokens,
        repetition_penalty=float(repetition_penalty),
    )
    Thread(target=model.generate, kwargs=generation_kwargs).start()
    for new_text in streamer:
        choice_data = ChatCompletionResponseStreamChoice(
            index=0, delta=DeltaMessage(content=new_text), finish_reason=None
        )
        chunk = ChatCompletionResponse(
            model=model_id, choices=[choice_data], object="chat.completion.chunk"
        )
        yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    choice_data = ChatCompletionResponseStreamChoice(
        index=0, delta=DeltaMessage(), finish_reason="stop"
    )
    chunk = ChatCompletionResponse(
        model=model_id, choices=[choice_data], object="chat.completion.chunk"
    )
    yield "{}".format(chunk.json(exclude_unset=True, ensure_ascii=False))
    yield "[DONE]"


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Creates a completion for the chat message"""
    msgs = request.messages
    print(msgs)
    # 注意数组越界问题
    print(time.asctime())
    lastmsg = msgs[-1]['content']
    qtype = classifier(lastmsg)
    print(qtype)

    # TODO： 拿着最后一条消息去封装classifier predict出分类结果
    # 如果 qtype
    # chat： 直接往下走
    # staticQ: 选关键词 选三元组
    # dynamicQ: 直接往下道歉
    # 是不是应该在predict里面加一个参数，把systemprompt带进去？ ////DONE
    # 对于不同的类型， 传入不同的prompt 那是不是应该额外写一个构造prompt的函数？
    if operator.contains(qtype, "static") or operator.contains(qtype, "静"):
        # Q:xxx
        # A:xxxx last answer的信息必然和上个Q强相关，且包含上个Q的必要信息 last q里面基本不可能越过上个QA去追溯到更早的QA中的实体 因为正常人不那么说话
        # Q:xxx
        global lastanswer
        if len(msgs) > 1:
            lastasnwer = msgs[-2]['content']
        else:
            lastasnwer = ""
        chathis = lastasnwer + lastmsg
        print(chathis)
        keyword = get_entity(chathis)
        print(keyword)
        tuple = retrieve_tuple(keyword)
        print(tuple)
        # TODO：把tuple安进prompt里面
        prompt_tuple = generate_chat_prompt_with_tuple(msgs, tuple)
        if isinstance(msgs, str):
            msgs = [ChatMessage(role="user", content=msgs)]
        else:
            msgs = [ChatMessage(role=x["role"], content=x["content"]) for x in msgs]
        if request.stream:
            generate = stream_predict(
                input=msgs,
                prompt=prompt_tuple,
                max_new_tokens=request.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                temperature=request.temperature,
                num_beams=request.num_beams,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
            )
            return EventSourceResponse(generate, media_type="text/event-stream")
        output = predict(
            input=msgs,
            prompt=prompt_tuple,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        choices = [
            ChatCompletionResponseChoice(index=i, message=msg) for i, msg in enumerate(msgs)
        ]
        choices += [
            ChatCompletionResponseChoice(
                index=len(choices), message=ChatMessage(role="assistant", content=output)
            )
        ]
        print(time.asctime())
        return ChatCompletionResponse(choices=choices)


    if operator.contains(qtype, "dynamic") or operator.contains(qtype, "动"):
        # 直接道歉
        prompt_aplgs = generate_chat_prompt_apologize(msgs)
        if isinstance(msgs, str):
            msgs = [ChatMessage(role="user", content=msgs)]
        else:
            msgs = [ChatMessage(role=x["role"], content=x["content"]) for x in msgs]
        if request.stream:
            generate = stream_predict(
                input=msgs,
                prompt=prompt_aplgs,
                max_new_tokens=request.max_tokens,
                top_p=request.top_p,
                top_k=request.top_k,
                temperature=request.temperature,
                num_beams=request.num_beams,
                repetition_penalty=request.repetition_penalty,
                do_sample=request.do_sample,
            )
            return EventSourceResponse(generate, media_type="text/event-stream")
        output = predict(
            input=msgs,
            prompt=prompt_aplgs,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        choices = [
            ChatCompletionResponseChoice(index=i, message=msg) for i, msg in enumerate(msgs)
        ]
        choices += [
            ChatCompletionResponseChoice(
                index=len(choices), message=ChatMessage(role="assistant", content=output)
            )
        ]
        print(time.asctime())
        return ChatCompletionResponse(choices=choices)


    # DEFAULT:qtype == chat
    if isinstance(msgs, str):
        msgs = [ChatMessage(role="user", content=msgs)]
    else:
        msgs = [ChatMessage(role=x["role"], content=x["content"]) for x in msgs]
    if request.stream:
        generate = stream_predict(
            input=msgs,
            prompt=None,
            max_new_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k,
            temperature=request.temperature,
            num_beams=request.num_beams,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
        )
        return EventSourceResponse(generate, media_type="text/event-stream")
    output = predict(
        input=msgs,
        prompt=None,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        num_beams=request.num_beams,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    )
    choices = [
        ChatCompletionResponseChoice(index=i, message=msg) for i, msg in enumerate(msgs)
    ]
    choices += [
        ChatCompletionResponseChoice(
            index=len(choices), message=ChatMessage(role="assistant", content=output)
        )
    ]
    print(time.asctime())
    return ChatCompletionResponse(choices=choices)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    """Creates a completion"""
    output = predict(
        input=request.prompt,
        prompt=None,
        max_new_tokens=request.max_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
        temperature=request.temperature,
        num_beams=request.num_beams,
        repetition_penalty=request.repetition_penalty,
        do_sample=request.do_sample,
    )
    choices = [CompletionResponseChoice(index=0, text=output)]
    return CompletionResponse(choices=choices)


if __name__ == "__main__":
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"][
        "fmt"
    ] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(app, host="0.0.0.0", port=19327, workers=1, log_config=log_config)

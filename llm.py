# by:Snowkingliu
# 2024/6/22 15:58
import os

import torch
from huggingface_hub import snapshot_download
from langchain.llms.base import LLM
from typing import Any, List, Optional
from langchain.callbacks.manager import CallbackManagerForLLMRun

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import pandas as pd

model_path = "internlm/internlm2-chat-1_8b"

# model_path = "models/Shanghai_AI_Laboratory/internlm2-chat-1_8b"
# model_dir = snapshot_download(
#     "Shanghai_AI_Laboratory/internlm2-chat-1_8b", cache_dir="models"
# )

base_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if torch.cuda.is_available():
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True
    ).cuda()
else:
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float,
        trust_remote_code=True,
    )
base_model = base_model.eval()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class PandasLLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModel = None

    def __init__(self, model, tokenizer):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        if not self.model:
            return "No model"
        # 重写调用函数
        system_prompt = """
         你是一个交通大模型，你有出租车的行程数据，数据存在pandas 的 df 中，
         你可以生成Python代码来将结果保存到 res 中，不要输出其他的语句，仅返回局部给 res 赋值部分代码即可
         df 的样例为：{"taxi_id":{"0":100320497,"1":100320497,"2":100320497,"3":100320497,"4":100320497},"trip_id":{"0":"100320497_1_150103_1","1":"100320497_1_150103_1","2":"100320497_1_150103_1","3":"100320497_1_150103_1","4":"100320497_1_150103_1"},"latitude":{"0":45.728577019,"1":45.7287195027,"2":45.729018556,"3":45.7287094957,"4":45.7293542663},"longitude":{"0":126.6871854236,"1":126.6869307195,"2":126.6864694495,"3":126.6869257035,"4":126.6868907077},"timestamp":{"0":1420245002,"1":1420245062,"2":1420245122,"3":1420245182,"4":1420245237},"idx":{"0":0,"1":1,"2":2,"3":3,"4":4}}
         比如，问你出租车的总数是多少，你就返回「```python\nres = df.taxi_id.drop_duplicates().count()\n```」，
         问你打印接单前十的出租车及接单数量，你就返回：「```python\nres = df[df.idx == 0].taxi_id.value_counts()[:10]\n```」
         返回的结果赋值给 res
         """

        messages = [(system_prompt, "")]
        response, history = self.model.chat(self.tokenizer, prompt, history=messages)
        return response

    @property
    def _llm_type(self) -> str:
        return "PandasLLM"


class TaxiLLM(LLM):
    # 基于本地 InternLM 自定义 LLM 类
    tokenizer: AutoTokenizer = None
    model: AutoModel = None

    def __init__(self, model, tokenizer):
        # model_path: InternLM 模型路径
        # 从本地初始化模型
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        if not self.model:
            return "No model"
        # 重写调用函数
        system_prompt = """
         你是一个交通大模型，你有出租车的行程数据，数据存在pandas 的 df 中，你可以根据上下文的文件信息，做出简要的回答
         我会先用 Python 执行一段 Python 代码，并且得到结果，你根据我的代码和结果做出做出判断
         只需简要回答问题，不用评价我的 Python 代码和结果，也不用说明我的 Python 代码的结果，直接回答答案
         模板如下：
         上下文: code: {Python 代码}, res: {该代码的返回}
         问题：{问题}
         """

        messages = [(system_prompt, "")]
        response, history = self.model.chat(self.tokenizer, prompt, history=messages)
        return response

    @property
    def _llm_type(self) -> str:
        return "TaxiLLM"


pandas_llm = PandasLLM(base_model, base_tokenizer)
taxi_llm = TaxiLLM(base_model, base_tokenizer)

df = pd.read_csv("./data/csv/nodes_lite.csv")


def execute(code):
    var = {}
    exec(code, {"df": df}, var)
    return var["res"]


def chat(question):
    r = pandas_llm(question)
    print(f"raw code: {r}")
    code = r.replace("```python", "```").split("```")[1].strip()
    p = execute(code)

    nq = f"""
    上下文: code: {code}, res: {p}
    问题：{question}
    """
    return taxi_llm(nq)

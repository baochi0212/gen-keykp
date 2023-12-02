import argparse
import os
import os.path as osp
import re
from nltk.stem.porter import *

from warnings import filterwarnings
filterwarnings("ignore")
import torch
# from tqdm import tqdm

import config
from dataset import load_data, load_dataset
from models.utils import load_config, load_tokenizer, load_model
from logger import FileLogger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import transformers
import torch
import os
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm

def handle_exception(string):
    present_keyphrase = string.split("Present keyphrase: ")[1].split("Absent keyphrase:")[0].strip()
    absent_keyphrase = string.split("Absent keyphrase: ")[1].strip()
    present_keyphrase = present_keyphrase.replace(",", ";") + ";"
    absent_keyphrase = absent_keyphrase.replace(",", ";") + ";"
    # present_keyphrase_list = [phrase.strip() for phrase in present_keyphrase.split(",")]
    # absent_keyphrase_list = [phrase.strip() for phrase in absent_keyphrase.split(",")]

    return present_keyphrase, absent_keyphrase
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sampling_params = SamplingParams(temperature=1e-10, max_tokens=1024)
# llm = LLM(model="/home/speechmt/chitb/SimCKP/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93")
llm = LLM(model="/home/speechmt/.cache/huggingface/hub/models--HuggingFaceH4--zephyr-7b-beta/snapshots/8af01af3d4f9dc9b962447180d6d0f8c5315da86")

tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta", token="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")
def make_simple_prompt(documents, tokenizer):
    chats = []
    PROMPT_TEMPLATE = """extract present and absent keyphrase in the given document. Given that present keyphrases are phrases that appear in the Document and absent not appear in Document. Both absent and present keyphrases represent Document's main idea.

Document: virtually enhancing the perception of user actions. This paper proposes using virtual reality to enhance the perception of actions by distant users on a shared application. Here, distance may refer either to space ( e.g. in a remote synchronous collaboration) or time ( e.g. during playback of recorded actions). Our approach consists in immersing the application in a virtual inhabited 3D space and mimicking user actions by animating avatars. We illustrate this approach with two applications, the one for remote collaboration on a shared application and the other to playback recorded sequences of user actions. We suggest this could be a low cost enhancement for telepresence.
Below are keyphrases capturing document's information: 
Present keyphrase: telepresence;animating;avatars
Absent keyphrase: application sharing;collaborative virtual environments

Document: {}
Below are keyphrases capturing document's information:"""
    for document in documents:
        prompt = PROMPT_TEMPLATE.format(document)
        chat = [
        {'role': 'user', 'content': prompt}
        ]

        tokenizer.use_default_system_prompt = False
        chat = tokenizer.apply_chat_template(chat, tokenize=False)
        chats.append(chat)
    return chats
def vllm_generate(llm, chats):
    outputs = llm.generate(chats, sampling_params=sampling_params)
    print("Finished batch inference!!!")
    for output in outputs:
        try:
            output = output.outputs[0].text.strip()
            output = output.split("<|assistant|>\n")[1].split("\n\nDocument")[0].split("\n\nBoth")[0]
        # print(handle_exception(output))
        # try:
        #     present, absent = output.split("\n")
        #     present = present.split(":")[1].strip().replace(",", ";") + ";"
        #     absent = absent.split(":")[1].strip().replace(",", ";") + ";"
        #     present_writer.write(present + '\n')
        #     abs_writer.write(absent + '\n')
        #     # print("????",present, absent)

            present, absent = handle_exception(output)
            present_writer.write(present + '\n')
            abs_writer.write(absent + '\n')
            print("???", present, absent)

        except:
            present_writer.write("BUG" + output + '\n')
            abs_writer.write("BUG" + output + '\n')

for dataset in ["krapivin", "nus", "semeval", "kp20k"]:
    present_writer = open(f"./zephyr_{dataset}_pre.txt", "a")
    abs_writer = open(f"./zephyr_{dataset}_abs.txt", "a")
    passed_line = len(open(f"./zephyr_{dataset}_pre.txt", "r").readlines())
    print("Passed", passed_line)
    documents = []
    for i, line in tqdm(enumerate(open(f"/home/speechmt/chitb/SimCKP/data/{dataset}/test_src.txt").readlines())):
        if i < passed_line:
            continue
        document = line.replace("<eos>", "").strip()
        documents.append(document)
    
    chats = make_simple_prompt(documents=documents, tokenizer=tokenizer)
    print("Batch prompt:", len(chats))
    vllm_generate(llm, chats)
    # break
    # break
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
# from models.utils import load_config, load_tokenizer, load_model
from logger import FileLogger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import transformers
import torch
import os
from vllm import LLM, SamplingParams
from tqdm.auto import tqdm
import sys
### --model_type --model_path

def handle_exception(string):
    # if 'Present keyphrases' in string:
    #     present_keyphrase = string.split("Present keyphrases:")[1].split("Absent keyphrases:")[0].strip()
    #     absent_keyphrase = string.split("Absent keyphrases:")[1].strip()
    # else:


    #     present_keyphrase = string.split("Present keyphrase: ")[1].split("Absent keyphrase:")[0].strip()
    #     absent_keyphrase = string.split("Absent keyphrase: ")[1].strip()
    present_keyphrase = string.split("Present:")[1].split("Absent:")[0].strip()
    absent_keyphrase = string.split("Absent:")[1].strip()
    present_keyphrase = present_keyphrase.replace(",", ";") + ";"
    absent_keyphrase = absent_keyphrase.replace(",", ";") + ";"
    # present_keyphrase_list = [phrase.strip() for phrase in present_keyphrase.split(",")]
    # absent_keyphrase_list = [phrase.strip() for phrase in absent_keyphrase.split(",")]

    return present_keyphrase, absent_keyphrase
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
sampling_params = SamplingParams(temperature=1e-10, max_tokens=1024)
model_path = sys.argv[2]
model_type = sys.argv[1]
if not os.path.exists(f"./llm_results/output_{model_type}"):
    os.mkdir(f"./llm_results/output_{model_type}")
# llm = LLM(model="/home/speechmt/chitb/SimCKP/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93")
#llm = LLM(model=model_path)
llm = LLM(model=model_path, gpu_memory_utilization=0.45)

tokenizer = AutoTokenizer.from_pretrained(model_path, token="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")
def make_simple_prompt(documents, tokenizer):
    chats = []
    PROMPT_TEMPLATE = """extract present and absent keyphrase in the given document. Given that present keyphrases are phrases that appear in the Document and absent not appear in Document. Both absent and present keyphrases represent Document's main idea.

Document: virtually enhancing the perception of user actions. This paper proposes using virtual reality to enhance the perception of actions by distant users on a shared application. Here, distance may refer either to space ( e.g. in a remote synchronous collaboration) or time ( e.g. during playback of recorded actions). Our approach consists in immersing the application in a virtual inhabited 3D space and mimicking user actions by animating avatars. We illustrate this approach with two applications, the one for remote collaboration on a shared application and the other to playback recorded sequences of user actions. We suggest this could be a low cost enhancement for telepresence.
Below are keyphrases capturing document's information: 
Present: telepresence;animating;avatars
Absent: application sharing;collaborative virtual environments

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

def make_finetune_prompt(documents):
    chats = []
    PROMPT_TEMPLATE = """### Instruction:
Extract present and absent keyphrases in the Input. Given that present keyphrases are phrases that appear in the Input and absent ones do not appear in Input. Both absent and present keyphrases represent main idea of Input's Document. 
### Input: {}
### Response:
"""
    for document in documents:
        chats.append(PROMPT_TEMPLATE.format(document))
    return chats
def vllm_generate(llm, chats):
    outputs = llm.generate(chats, sampling_params=sampling_params)
    print("Number of outputs: ", len(outputs))
    presents = []
    absents = []
    bugs = []
    for i, og_output in enumerate(outputs):
        try:
            output = og_output.outputs[0].text.strip()
            if "finetune" in model_type:
                output = output.split("### End")[0]
            elif "zephyr" in model_type:
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
            present_writer.write(present.strip() + '\n')
            abs_writer.write(absent.strip() + '\n')
            presents.append(present)
            absents.append(absent)
            # print("???", present, absent)

        except:
            bugs.append(1)
            print("bug: ", output)
            present_writer.write("BUG" + og_output.outputs[0].text.strip() + '\n')
            abs_writer.write("BUG" + og_output.outputs[0].text.strip() + '\n')
    print("Final write", len(presents), len(absents), len(bugs))

for dataset in ["inspec", "krapivin", "nus", "semeval"]:
# for dataset in ["inspec"]:
    present_writer = open(f"./llm_results/output_{model_type}/{model_type}_{dataset}_pre.txt", "w")
    abs_writer = open(f"./llm_results/output_{model_type}/{model_type}_{dataset}_abs.txt", "w")
    passed_line = len(open(f"./llm_results/output_{model_type}/{model_type}_{dataset}_pre.txt", "r").readlines())
    print("Passed", passed_line)
    documents = []
    for i, line in tqdm(enumerate(open(f"./data/{dataset}/test_src_filtered_bart.txt").readlines())):
        if i < passed_line:
            continue
        document = line.replace("<eos>", "").strip()
        documents.append(document)
    
    chats = make_finetune_prompt(documents=documents)
    print("Batch prompt:", len(chats))
    
    bs = 500
    for i in range(0, len(chats), bs):
        end = i+bs if i+bs < len(chats) else len(chats)
        print("Number of prompts", len(chats[i:end]))
        vllm_generate(llm, chats[i:end])

    # print("Lines", len(present_reader.readlines()), len(abs_reader.readlines()))
    # break
    # break

from vllm import LLM, SamplingParams
import sys

## --model_path
model_path = sys.argv[1]
sampling_params = SamplingParams(temperature=1e-10, max_tokens=1024)
# llm = LLM(model="/home/speechmt/chitb/SimCKP/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93")
llm = LLM(model=model_path)

## mistral 7b
PROMPT_TEMPLATE = """extract present and absent keyphrase in the given document. Given that present keyphrases are phrases that appear in the Document and absent not appear in Document. Both absent and present keyphrases represent Document's main idea.

Document: virtually enhancing the perception of user actions. This paper proposes using virtual reality to enhance the perception of actions by distant users on a shared application. Here, distance may refer either to space ( e.g. in a remote synchronous collaboration) or time ( e.g. during playback of recorded actions). Our approach consists in immersing the application in a virtual inhabited 3D space and mimicking user actions by animating avatars. We illustrate this approach with two applications, the one for remote collaboration on a shared application and the other to playback recorded sequences of user actions. We suggest this could be a low cost enhancement for telepresence.
Below are keyphrases capturing document's information: 
Present: telepresence;animating;avatars
Absent: application sharing;collaborative virtual environments

Document: {}
Below are keyphrases capturing document's information:"""
document = "Subquadratic Approximation Algorithms for Clustering Problems in High Dimensional Spaces. <eos> One of the central problems in information retrieval, data mining, computational biology, statistical analysis, computer vision, geographic analysis, pattern recognition, distributed protocols is the question of classification of data according to some clustering rule. Often the data is noisy and even approximate classification is of extreme importance. The difficulty of such classification stems from the fact that usually the data has many incomparable attributes, and often results in the question of clustering problems in high dimensional spaces. Since they require measuring distance between every pair of data points, standard algorithms for computing the exact clustering solutions use quadratic or nearly quadratic running time&semi; i.e., O(dn2(d)) time where n is the number of data points, d is the dimension of the space and (d) approaches 0 as d grows. In this paper, we show (for three fairly natural clustering rules) that computing an approximate solution can be done much more efficiently. More specifically, for agglomerative clustering (used, for example, in the Alta Vista search engine), for the clustering defined by sparse partitions, and for a clustering based on minimum spanning trees we derive randomized (1 approximation algorithms with running times (d2 n2) where  > 0 depends only on the approximation parameter &epsi; and is independent of the dimension d."
prompt = PROMPT_TEMPLATE.format(document)
chat = [
{'role': 'user', 'content': prompt}
]

tokenizer.use_default_system_prompt = False
chat = tokenizer.apply_chat_template(chat, tokenize=False)
og_outputs = llm.generate(chat, sampling_params=sampling_params)[0]
output = og_outputs.outputs[0].text.strip()
print(output.split("<|assistant|>\n")[1])

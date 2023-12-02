from transformers import AutoModelForCausalLM
import inspect

model_path = "./models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93"
model = AutoModelForCausalLM.from_pretrained(model_path)
# Get the content of the function
function_content = inspect.getsource(model.save)

# Print the function content
print(function_content)

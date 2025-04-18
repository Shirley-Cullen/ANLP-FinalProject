import sys

import fire
import gradio as gr
import torch
torch.set_num_threads(1)
import transformers
import json
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from sklearn.metrics import roc_auc_score
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

try:
    if torch.backends.mps.is_available():
        device = "mps"
except:  # noqa: E722
    pass

def clean_ans(ans):
    ans = ans.split("Explanation")[0]
    ans = ans.split("\n")[0]
    # ans = ans.split("No")[0]
    # ans = ans.split("Yes")[0]
    return ans

def main(
    load_8bit: bool = False,
    base_model: str = "",
    lora_weights: str = "tloen/alpaca-lora-7b",
    test_data_path: str = "data/test.json",
    result_json_data: str = "temp.json",
    batch_size: int = 16,
    share_gradio: bool = False,
):
    assert (
        base_model
    ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

    model_type = lora_weights.split('/')[-1]
    model_name = '_'.join(model_type.split('_')[:2])

    if model_type.find('book') > -1:
        train_sce = 'book'
    else:
        train_sce = 'movie'
    
    if test_data_path.find('book') > -1:
        test_sce = 'book'
    else:
        test_sce = 'movie'
    temp_list = model_type.split('_')
    print(temp_list)
    seed = temp_list[-2]
    sample = temp_list[-1]
    
    if os.path.exists(result_json_data):
        f = open(result_json_data, 'r')
        data = json.load(f)
        f.close()
    else:
        data = dict()

    if not data.__contains__(train_sce):
        data[train_sce] = {}
    if not data[train_sce].__contains__(test_sce):
        data[train_sce][test_sce] = {}
    if not data[train_sce][test_sce].__contains__(model_name):
        data[train_sce][test_sce][model_name] = {}
    if not data[train_sce][test_sce][model_name].__contains__(seed):
        data[train_sce][test_sce][model_name][seed] = {}
    if not data[train_sce][test_sce][model_name][seed].__contains__(sample):
        data[train_sce][test_sce][model_name][seed][sample] = {}
        # exit(0)


    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    elif device == "mps":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )


    tokenizer.padding_side = "left"
    # unwind broken decapoda-research config
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    if not load_8bit:
        model.half()  # seems to fix bugs for some users.

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    def evaluate(
        instructions,
        inputs=None,
        temperature=0,
        top_p=1.0,
        top_k=40,
        num_beams=1,
        max_new_tokens=128,
        batch_size=1,
        **kwargs,
    ):
        prompt = [generate_prompt(instruction, input) for instruction, input in zip(instructions, inputs)]
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                # batch_size=batch_size,
            )
        s = generation_output.sequences
        scores = generation_output.scores[0].softmax(dim=-1)
        logits = torch.tensor(scores[:,[8241, 3782]], dtype=torch.float32).softmax(dim=-1)
        input_ids = inputs["input_ids"].to(device)
        L = input_ids.shape[1]
        s = generation_output.sequences
        output = tokenizer.batch_decode(s, skip_special_tokens=True)
        output = [_.split('Response:\n')[-1] for _ in output]
        output = [clean_ans(_) for _ in output]
        print(f"output: {output}")
        
        return output, logits.tolist()
        
    # testing code for readme
    logit_list = []
    gold_list= []
    outputs = []
    logits = []
    from tqdm import tqdm
    gold = []
    pred = []

    with open(test_data_path, 'r') as f:
        test_data = json.load(f)
        instructions = [_['instruction'] for _ in test_data]
        inputs = [_['input'] for _ in test_data]
        gold = [int(_['output'] == 'Yes.') for _ in test_data]
        def batch(list, batch_size=16):
            chunk_size = (len(list) - 1) // batch_size + 1
            for i in range(chunk_size):
                yield list[batch_size * i: batch_size * (i + 1)]
        for i, batch in tqdm(enumerate(zip(batch(instructions), batch(inputs)))):
            instructions, inputs = batch
            output, logit = evaluate(instructions, inputs)
            outputs = outputs + output
            logits = logits + logit
        for i, test in tqdm(enumerate(test_data)):
            test_data[i]['predict'] = outputs[i]
            test_data[i]['logits'] = logits[i]
            pred.append(logits[i][0])

    from sklearn.metrics import roc_auc_score

    # save metrics
    # data[train_sce][test_sce][model_name][seed][sample] = roc_auc_score(gold, pred)
    # print(f"hi: {data[train_sce][test_sce][model_name][seed].keys()}")
    data[train_sce][test_sce][model_name][seed][sample]["roc_auc"] = roc_auc_score(gold, pred)
    # evaluate explanation
    if "explanation" in test_data[0]:
        references = [f"{dp['output']} {dp['explanation']}" for dp in test_data]
        print("=" * 40)
        print("Compute BERT score for explanation")
        print(f"Prediction: {outputs[0]}")
        print(f"Reference: {references[0]}")
        print("=" * 40)
        from bert_score import score
        # Compute BERTScore (precision, recall, F1), based on list of predictions and list of references
        P, R, F1 = score(outputs, references, lang="en", rescale_with_baseline=True)
        P_, R_, F1_ = score(outputs, references, lang="en", rescale_with_baseline=False)
        for i, test in tqdm(enumerate(test_data)):
            test_data[i]['Precision'] = P[i].item()
            test_data[i]['Recall'] = R[i].item()
            test_data[i]['F1'] = F1[i].item()
            test_data[i]['Precision_origin'] = P_[i].item()
            test_data[i]['Recall_origin'] = R_[i].item()
            test_data[i]['F1_origin'] = F1_[i].item()
        data[train_sce][test_sce][model_name][seed][sample]["bert_precision"] = P.mean().item()
        data[train_sce][test_sce][model_name][seed][sample]["bert_recall"] = R.mean().item()
        data[train_sce][test_sce][model_name][seed][sample]["bert_f1"] = F1.mean().item()
        data[train_sce][test_sce][model_name][seed][sample]["bert_precision_origin"] = P_.mean().item()
        data[train_sce][test_sce][model_name][seed][sample]["bert_recall_origin"] = R_.mean().item()
        data[train_sce][test_sce][model_name][seed][sample]["bert_f1_origin"] = F1_.mean().item()
        print(f"Precision: {P.mean():.4f}")
        print(f"Recall:    {R.mean():.4f}")
        print(f"F1 Score:  {F1.mean():.4f}")
        print(data)
    f = open(result_json_data, 'w')
    json.dump(data, f, indent=4)
    f.close()

    # save output
    print(result_json_data)
    output_json_path = f"{result_json_data.split('.')[0]}_output.json"
    f = open(output_json_path, 'w')
    json.dump(test_data, f, indent=4)
    f.close()

def generate_prompt(instruction, input=None):
    if input:
        return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Input:
{input}

### Response:
"""
    else:
        return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.  # noqa: E501

### Instruction:
{instruction}

### Response:
"""


if __name__ == "__main__":
    fire.Fire(main)

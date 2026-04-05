import csv
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from enum import Enum


class Model(Enum):
    GPT_OSS = 1,
    PHI_THREE_MINI = 2


_fields = ['text', 'label']
_data_dict: list[dict[str, str]] = []
_filename: str = "/Users/paulpoomrit/1_SFU/8_Spring_2026/LING450_CompLing/LING450_TermProject/processed_csv/data.csv"
_ai_prompt = "Consider the following text and imagine you were a human prompter, trying to reverse engineer the prompt of such text. Provide the prompt: "
_format_prompt = "Generate text using HTML tags for formatting."


def get_ai_prompt_from_transformers(human_text: str, model_id: str) -> str:

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map = "auto",
        dtype = "auto",
        # offload_folder="offload",
        # trust_remote_code = True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    messages = [
        {"role": "user", "content": _ai_prompt + human_text},
    ]

    outputs = pipe(
        messages
    )

    return outputs[0]["generated_text"][-1] + _format_prompt


def get_ai_prompt(human_text: str, model_enum = Model) -> str:

    match model_enum:
        case Model.GPT_OSS:
            return get_ai_prompt_from_transformers(human_text, "openai/gpt-oss-20b")
        case Model.PHI_THREE_MINI:
            return get_ai_prompt_from_transformers(human_text, "microsoft/Phi-3-mini-4k-instruct")
        case _:
            return get_ai_prompt_from_transformers(human_text, "microsoft/Phi-3-mini-4k-instruct")


def get_ai_counterpart_from_transformers(human_text: str, model_id: str) -> str:
    prompt = get_ai_prompt(human_text, Model.PHI_THREE_MINI)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map = "auto",
        dtype = "auto",
        # offload_folder="offload",
        # trust_remote_code = True
    )

    tokenizer = AutoTokenizer.from_pretrained (model_id)

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer
    )

    messages = [
        {"role": "user", "content": prompt},
    ]

    outputs = pipe(
        messages
    )

    return outputs[0]["generated_text"][-1]


def get_ai_counterpart(human_text: str, model_enum = Model) -> str:

    ai_text = ""

    match model_enum:
        case Model.GPT_OSS:
            return get_ai_counterpart_from_transformers(human_text, "openai/gpt-oss-20b")
        case Model.PHI_THREE_MINI:
            return get_ai_counterpart_from_transformers(human_text, "microsoft/Phi-3-mini-4k-instruct")
        case _:
            return get_ai_counterpart_from_transformers(human_text, "microsoft/Phi-3-mini-4k-instruct")


def process_doc(document_path: str):
    with open(document_path, 'r') as file:
        for line in file:

            if line == "\n":
                continue

            human_dict: dict[str, str] = {}
            human_dict['text'] = line
            human_dict['label'] = 'H'
            _data_dict.append(human_dict)

            ai_dict: dict[str, str] = {}
            # TODO: replace with get_ai_counterpart(line)
            ai_dict['text'] = get_ai_counterpart(line)
            ai_dict['label'] = "AI"
            _data_dict.append(ai_dict)

            # TODO: update csv file here


def main():
    # TODO: replace this with a for-loop over data folder
    process_doc(
        "/Users/paulpoomrit/1_SFU/8_Spring_2026/LING450_CompLing/LING450_TermProject/data/coca-samples-text/text_acad.txt")

    with open(_filename, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=_fields)
        writer.writeheader()
        writer.writerows(_data_dict)


# main()
print(get_ai_counterpart("We live in capitalism, its power seems inescapable — but then, so did the divine right of kings. Any human power can be resisted and changed by human beings. Resistance and change often begin in art. Very often in our art, the art of words.",Model.PHI_THREE_MINI))
import csv
from transformers import pipeline
import torch
from enum import Enum


class Model(Enum):
    GPT_OSS = 1


_fields = ['text', 'label']
_data_dict: list[dict[str, str]] = []
_filename: str = "/Users/paulpoomrit/1_SFU/8_Spring_2026/LING450_CompLing/LING450_TermProject/processed_csv/data.csv"
_ai_prompt = "Consider the following text and imagine you were a human prompter, trying to reverse engineer the prompt of such text. Provide the prompt: "


def get_ai_prompt_from_gpt_oss(human_text: str) -> str:
    model_id = "openai/gpt-oss-120b"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
    )

    messages = [
        {"role": "user", "content": _ai_prompt + human_text},
    ]

    outputs = pipe(
        messages
    )

    return outputs[0]["generated_text"][-1]


def get_ai_prompt(human_text: str, model_enum = Model) -> str:

    prompt = ""

    match model_enum:
        case Model.GPT_OSS:
            prompt = get_ai_prompt_from_gpt_oss(human_text)
        case _:
            prompt = get_ai_prompt_from_gpt_oss(human_text)

    return prompt


def get_ai_counterpart_from_gpt_oss(human_text: str) -> str:
    prompt = get_ai_prompt(human_text)

    model_id = "openai/gpt-oss-120b"

    pipe = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype="auto",
        device_map="auto",
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
            ai_text = get_ai_counterpart(human_text)
        case _:
            ai_text = get_ai_counterpart(human_text)

    return ai_text


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

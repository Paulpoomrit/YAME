import csv
from openai import OpenAI


_fields = ['text', 'label']
_data_dict: list[dict[str, str]] = []
_filename: str = "/Users/paulpoomrit/1_SFU/8_Spring_2026/LING450_CompLing/LING450_TermProject/processed_csv/data.csv"
_open_ai_key = "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"
_ai_prompt = "Consider the following text and imagine you were a human prompter, trying to reverse engineer the prompt of such text. Provide the prompt: "


client = OpenAI(api_key=_open_ai_key)


def get_ai_prompt(human_text: str) -> str:
    messages = [{"role": "system", "content":
                  "You are a intelligent assistant."}]
    messages.append(
        {"role": "user", "content": _ai_prompt+human_text},
    )
    chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return chat.choices[0].message.content


def get_ai_counterpart(human_text: str) -> str:
    prompt = get_ai_prompt(human_text)
    messages = [{"role": "system", "content":
                  "You are a intelligent assistant."}]
    messages.append(
        {"role": "user", "content": prompt},
    )
    chat = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
    return chat.choices[0].message.content


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
            ai_dict['text'] = "yo" # TODO: replace with get_ai_counterpart(line)
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


main()
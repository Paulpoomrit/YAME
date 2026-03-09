from enum import Enum
import re

class Feature(Enum):
    # --------- Group A: Features related to language, grammar and style --------- #
    EMDASH = 1
    EMOJI = 2
    BOLD = 3
    TITLE_CASE = 4
    LIST = 5
    CURLY_QUOTATION = 6
    TABLE = 7
    TEMPLATE = 8
    GRAMMAR = 9
    VOCAB = 10
    TRANSITION_WORD = 11
    # ------------------- Group B: Features related to content ------------------- #
    NEG_PARALLELISM = 12
    RULE_OF_THREE = 13
    BASIC_COPULATIVE = 14
    ELEG_VARIATION = 15
    # ------------------------- Group C: Binary features ------------------------- #
    SUBJECT_LINE = 16
    COMMUNICATION = 17
    KNOWLEDGE_CUTOFF = 18
    SUMMARY = 19


regex_dict: dict[Feature, str] = {
    Feature.EMDASH: r"—",
    Feature.EMOJI: r"/(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])/g",
    Feature.BOLD: r"<(b|strong|i|em)>.*[^>]<\/(b|strong|i|em)>",
    Feature.TITLE_CASE: r"^(?:[A-Z][^\s]*\s?)+$",
    Feature.LIST: r"<ul>|<ol>",
    Feature.CURLY_QUOTATION: r"‘|“",
    Feature.TABLE: r"<table>",
    Feature.TEMPLATE: r"\[.+\]"
}


ai_vocab = ["additionally","align with","boasts","bolstered",
            "crucial","delve","emphasizing","enduring","enhance"
            "fostering","garner","highlight","interplay","intricate",
            "intricacies","key","landscape","meticulous","meticulously",
            "pivotal","showcase","tapestry","testament","underscore"
            "valuable","vibrant"]


def count_all_words_in_list(text: str, words: list[str]):
    total_count: int = 0

    text = text.lower()
    for word in words:
        word = word.lower()
        match_list = re.findall(word, text)
        total_count += len(match_list)

    return

def extract_group_a(text: str, feature_type: Feature, total_num_words: int):
    regex = regex_dict[feature_type]
    match_list = re.findall(regex, text)
    return len(match_list)























def extract_features(document_path: str):
   with open(document_path, 'r') as file:
       for line in file:
           print(line)
           print('-----')

extract_features("/Users/paulpoomrit/1_SFU/8_Spring_2026/LING450_CompLing/LING450_TermProject/data/coca-samples-text/text_acad.txt")
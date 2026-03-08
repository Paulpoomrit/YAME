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
    VOCAB = 8
    NEG_PARALLELISM = 9
    TRANSITION_WORD = 10
    GRAMMAR = 11
    TEMPLATE = 12
    # ------------------- Group B: Features related to content ------------------- #
    RULE_OF_3 = 13
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
    Feature.TITLE_CASE: r"^(?:[A-Z][^\s]*\s?)+$"
}


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
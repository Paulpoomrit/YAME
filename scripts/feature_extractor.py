from enum import Enum
import re
import language_tool_python
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')

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
    # ------------------- Group B: Features related to content ------------------- #
    NEG_PARALLELISM = 11
    RULE_OF_THREE = 12 #TODO
    BASIC_COPULATIVE = 13 #TODO
    ELEG_VARIATION = 14 #TODO
    # ------------------------- Group C: Binary features ------------------------- #
    SUBJECT_LINE = 15 #TODO
    COMMUNICATION = 16 #TODO
    KNOWLEDGE_CUTOFF = 17 #TODO
    SUMMARY = 18 #TODO


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

    return total_count


def count_all_grammartical_mistakes(text: str) -> int:
    with language_tool_python.LanguageTool("en-US") as tool:
        matches = tool.check(text)
        return len(matches)
    return -1


def negative_parallelism_ratio(text: str) -> int:
    """ Returns the ratio of sentences with negative parallelism to
    all sentences with negative sentiment
    (that does or does not come with its parallel counterpart). """

    text = text.lower()
    neg_paralleism_match = re.findall(r"(?:(?:\bnot\b|\bno\b){1}|\bain['’]t\b|\bisn['’]t\b|\bis not\b)(?:.*?(?:\bbut\b|\bhowever\b|\brather\b|\bis\b|['’]s))", text)
    all_neg_match = re.findall(r"(?:(?:\bnot\b|\bno\b){1}|\bain['’]t\b|\bisn['’]t\b|\bis not\b)", text)

    if len(neg_paralleism_match) == 0:
        return 0

    return len(neg_paralleism_match)/ len(all_neg_match)


def rule_of_three_ratio(text: str) -> int:
    total_num_senteces = 0
    total_rule_of_three_sentences = 0
    sentences = nltk.sent_tokenize(text)
    rule_of_three_regex = re.compile(r"\b(\w+)\b,\s(.+),\s(.+)")

    for sentence in sentences:
        total_num_senteces += 1
        # print(sentence)
        match = rule_of_three_regex.findall(sentence)


        if len(match) > 0:
            match_list = list(match[0])
            tagged = nltk.pos_tag(match_list, tagset='universal')
            # print(tagged)

            # Two of the examples has to be of the same type to qualify
            if (tagged[0][1] == tagged[1][1] or tagged[1][1] == tagged[2][1] or tagged[0][1] == tagged[2][1]):
                total_rule_of_three_sentences += 1

    if total_num_senteces == 0 or total_rule_of_three_sentences == 0:
        return 0

    return total_rule_of_three_sentences/total_num_senteces


def extract_group_a(text: str, feature_type: Feature, total_num_words: int):

    if feature_type == Feature.VOCAB:
        return count_all_words_in_list(text, ai_vocab) / total_num_words

    regex = regex_dict[feature_type]
    match_list = re.findall(regex, text)
    return len(match_list) / total_num_words


def extract_features(document_path: str):
   with open(document_path, 'r') as file:
       for line in file:
           # TODO: extract feature here
           print(line)
           print('-----')

## Test
#extract_features("/Users/paulpoomrit/1_SFU/8_Spring_2026/LING450_CompLing/LING450_TermProject/data/coca-samples-text/text_acad.txt")
#print(count_all_grammartical_mistakes("I is Paul."))
# print(negative_parallelism_ratio("We live in capitalism, its power seems inescapable — but then, so did the divine right of kings. Any human power can be resisted and changed by human beings. Resistance and change often begin in art. Very often in our art, the art of words."))
# print(negative_parallelism_ratio("a Not just x but y. ain't x rather k. That’s not just a sourcing issue—it's a systemic bias. The issue here isn't just sourcing—it's framing. The issue here is not just sourcing—it's framing. Constitutes not only a work of self-representation, but a visual document. huh"))

print(rule_of_three_ratio("This is so insane, right brother? The Amaze Conference brings together global SEO professionals, marketing experts, and growth hackers to discuss the latest trends in digital marketing. The event features keynote sessions, panel discussions, and networking opportunities."))
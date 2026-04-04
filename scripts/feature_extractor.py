from scipy import spatial
import gensim.downloader as api
from enum import Enum
import re
import language_tool_python
import nltk
import numpy as np
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('universal_tagset')

# for word2vec
word_2_vec = api.load("glove-wiki-gigaword-50")


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
    RULE_OF_THREE = 12
    BASIC_COPULATIVE = 13
    ELEG_VARIATION = 14
    # ------------------------- Group C: Binary features ------------------------- #
    SUBJECT_LINE = 15  # TODO
    COMMUNICATION = 16  # TODO
    KNOWLEDGE_CUTOFF = 17  # TODO
    SUMMARY = 18  # TODO


regex_dict: dict[Feature, str] = {
    Feature.EMDASH: r"—",
    Feature.EMOJI: r"/(\u00a9|\u00ae|[\u2000-\u3300]|\ud83c[\ud000-\udfff]|\ud83d[\ud000-\udfff]|\ud83e[\ud000-\udfff])/g",
    Feature.BOLD: r"<(b|strong|i|em)>.*[^>]<\/(b|strong|i|em)>",
    Feature.TITLE_CASE: r"^(?:[A-Z][^\s]*\s?)+$",
    Feature.LIST: r"<ul>|<ol>",
    Feature.CURLY_QUOTATION: r"‘|“",
    Feature.TABLE: r"<table>",
    Feature.TEMPLATE: r"\[.+\]",
    Feature.SUBJECT_LINE: r"subject: "
}


ai_vocab = ["additionally", "align with", "boasts", "bolstered",
            "crucial", "delve", "emphasizing", "enduring", "enhance"
            "fostering", "garner", "highlight", "interplay", "intricate",
            "intricacies", "key", "landscape", "meticulous", "meticulously",
            "pivotal", "showcase", "tapestry", "testament", "underscore"
            "valuable", "vibrant"]

summary_words = ["in summary", "in conclusion", "overall", "to conclude", "to put it briefly",
                 "all things considered", "as a final point", "in a nutshell", "the long and the short of it",
                 "finally", "lastly", "ultimately"]


def count_all_words_in_list(text: str, words: list[str]) -> int:
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


def negative_parallelism_ratio(text: str) -> float:
    """ Returns the ratio of sentences with negative parallelism to
    all sentences with negative sentiment
    (that does or does not come with its parallel counterpart). """

    text = text.lower()
    neg_paralleism_match = re.findall(
        r"(?:(?:\bnot\b|\bno\b){1}|\bain['’]t\b|\bisn['’]t\b|\bis not\b)(?:.*?(?:\bbut\b|\bhowever\b|\brather\b|\bis\b|['’]s))", text)
    all_neg_match = re.findall(
        r"(?:(?:\bnot\b|\bno\b){1}|\bain['’]t\b|\bisn['’]t\b|\bis not\b)", text)

    if len(neg_paralleism_match) == 0:
        return 0

    return len(neg_paralleism_match) / len(all_neg_match)


def rule_of_three_ratio(text: str) -> float:
    """ Returns the ratio of sentences, giving three examples,
    divided by the total number of all sentences.
    his is being extracted by considering the form
    that such a sentence usually presents itself in,
    that is, three words/phrases mediated by “,” in the middle. """

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


def basic_copulative_ratio(text: str) -> float:
    """  Returns the ratio of basic copulative words
    to the number of all white-space separated tokens. """

    word_tokens = nltk.word_tokenize(text)
    total_num_tokens = len(word_tokens)

    if total_num_tokens == 0:
        return 0

    basic_cop_pattern = re.compile(
        r"\bis\b|\bare\b|\bam\b|\bbeing\b|\bwas\b|\bwere\b|\bbeen\b|\bbe\b")
    match = basic_cop_pattern.findall(text)
    num_basic_cop = len(match)

    if num_basic_cop == 0:
        return 0

    return num_basic_cop/total_num_tokens


def eleg_variation_ratio(text: str) -> str:

    words = nltk.word_tokenize(text)
    word_matrix = []

    for word in words:
        word = word.lower()
        word_vec = word_2_vec[word]  # grab word vector
        word_matrix.append(word_vec)

    # Computes the cosine distance between each pair of the word vector
    pairwise_dist = spatial.distance.pdist(word_matrix, metric='cosine')
    avg = np.average(pairwise_dist)

    return avg


def is_there_the_following_pattern(pattern: str, text: str) -> bool:

    subject_line_regex = re.compile(pattern, flags=re.IGNORECASE)
    match = re.findall(subject_line_regex, text)

    if len(match) > 0:
        return True
    else:
        return False


def extract_group_a(text: str, feature_type: Feature, total_num_words: int):

    if feature_type == Feature.VOCAB:
        return count_all_words_in_list(text, ai_vocab) / total_num_words

    regex = regex_dict[feature_type]
    match_list = re.findall(regex, text)
    return len(match_list) / total_num_words


def if_words_exist(text: str, word_list: list[str]) -> bool:
    for word in word_list:
        word = word.lower()
        match_list = re.findall(word, text, flags=re.IGNORECASE)
        if len(match_list) >= 1:
            return True

    return False


def extract_features(document_path: str):
    with open(document_path, 'r') as file:
        for line in file:
            # TODO: extract feature here
            print(line)
            print('-----')

# Test
# extract_features("/Users/paulpoomrit/1_SFU/8_Spring_2026/LING450_CompLing/LING450_TermProject/data/coca-samples-text/text_acad.txt")
# print(count_all_grammartical_mistakes("I is Paul."))
# print(negative_parallelism_ratio("We live in capitalism, its power seems inescapable — but then, so did the divine right of kings. Any human power can be resisted and changed by human beings. Resistance and change often begin in art. Very often in our art, the art of words."))
# print(negative_parallelism_ratio("a Not just x but y. ain't x rather k. That’s not just a sourcing issue—it's a systemic bias. The issue here isn't just sourcing—it's framing. The issue here is not just sourcing—it's framing. Constitutes not only a work of self-representation, but a visual document. huh"))

# print(rule_of_three_ratio("This is so insane, right brother? The Amaze Conference brings together global SEO professionals, marketing experts, and growth hackers to discuss the latest trends in digital marketing. The event features keynote sessions, panel discussions, and networking opportunities."))
# print(basic_copulative("Gallery 825 on [[La Cienega Boulevard]], which was purchased in 1958, is LAAA's exhibition arm for [[contemporary art]]. There are four individual gallery spaces[...]"))
# print(basic_copulative("Gallery 825 on [[La Cienega Boulevard]] serves as LAAA's exhibition space for contemporary art. The gallery features four separate spaces[...]"))

# print(eleg_variation_ratio("animal creature beast"))
# print(eleg_variation_ratio("Vierny, after a visit in Moscow in the early 1970’s, committed to supporting artists resisting the constraints of socialist realism and discovered Yankilevskly, among others such as Ilya Kabakov and Erik Bulatov. In the challenging climate of Soviet artistic constraints, Yankilevsky, alongside other non-conformist artists, faced obstacles in expressing their creativity freely. Dina Vierny, recognizing the immense talent and the struggle these artists endured, played a pivotal role in aiding their artistic aspirations. [...]"))


# print(is_there_the_following_pattern(regex_dict[Feature.SUBJECT_LINE], "Subject: Request for Permission to Edit Wikipedia Article"))
# print(is_there_the_following_pattern(regex_dict[Feature.SUBJECT_LINE], "Request for Permission to Edit Wikipedia Article"))


print(if_words_exist("the educational and training trajectory for nurse scientists typically involves a progression from a master's degree in nursing to a Doctor of Philosophy in Nursing, followed by postdoctoral training in nursing research. This structured pathway ensures that nurse scientists acquire the necessary knowledge and skills to engage in rigorous research and contribute meaningfully to the advancement of nursing science.", summary_words))
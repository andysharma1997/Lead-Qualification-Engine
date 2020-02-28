import re


def find_questions(test_str):
    quest = []

    # find question mark and the complete sentence
    regex = r"(^|(?<=[.?!]))\s*[A-Za-z,;'\"\s]+\?"
    matches = re.finditer(regex, test_str, re.IGNORECASE)
    for match in matches:
        quest.append(match.group())

    # find 5W 1H and the complete sentence
    whQue = re.findall(r"(how|can|what|where|describe|who|when|why)(?i)", test_str)
    for que in whQue:
        regex = r"[^.?!]*(?<=[.?\s!])" + que + "(?=[\s.?!])[^.?!]*[.?!]"
        test_str = "." + test_str + "."
        result = re.findall(regex, test_str)
        quest = quest + result
    # return list(set(quest))
    return list(set([item.strip() for item in quest]))

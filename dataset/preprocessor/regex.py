import re

# 여기에 각자 작성한 regex 및 이를 활용한 전처리기 만들어서 활용해주세요~
# 기존에 작성된건 수정하지 말고 계속 추가해주시면 될 것 같아요!!!
class Regex:
    pattern_remove_duplicated_words = re.compile(r'\b(\w+)( \1\b)+')
    # reference: https://stackoverflow.com/questions/17238587/python-regular-expression-to-remove-repeated-words

    def remove_duplicated_words(txt):
        # 연속으로 중복된 단어를 제거합니다
        # 다만, 중복된 2단어 이상의 '구'는 제거하지 못합니다
        return Regex.pattern_remove_duplicated_words.sub(r'\1', txt)

# 실험 예시
txt = "this is a duplicated duplicated duplicated word from an error occured occured error occured"
print(Regex.remove_duplicated_words(txt))
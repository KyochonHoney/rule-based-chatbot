from kiwipiepy import Kiwi

kiwi = Kiwi()

def tokenize_ko(text: str) -> str:
    # 형태소 표층형 중 명사/동사어간/형용사어간 위주 추출(간단 예시)
    toks = []
    for sent in kiwi.tokenize(text):
        for t in sent:
            pos = t.tag.split("/")[0]
            if pos in ("NNG","NNP","VV","VA","MAG","SL"):
                toks.append(t.form)
    return " ".join(toks)

# 실제 색인 로직은 app/main.py의 /index 엔드포인트에서 호출됩니다.
# 이 파일은 토크나이저 함수를 제공하는 역할을 합니다.

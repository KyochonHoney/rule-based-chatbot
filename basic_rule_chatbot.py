import pandas as pd

chatbot_data = pd.read_excel("./chatbot_data.xlsx")
#print(chatbot_data)

chat_dic = {}
row = 0
for rule in chatbot_data['rule']:
    chat_dic[row] = rule.split('|')
    row += 1

#print(chat_dic)

def chat(request):
    for k, v in chat_dic.items():
        chat_flag = False
        for word in v:
            if word in request:
                chat_flag = True
            else:
                chat_flag = False
                break
        if chat_flag:
            return chatbot_data['response'][k]
    return '무슨 말인지 모르겠습니다. 다시 질문해주세요'

while True:
    req = input('대화를 입력해보세요. ')
    if req == 'exit':
        break
    else:
        print('챗봇:', chat(req))
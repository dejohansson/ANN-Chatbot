from chatbot import core as cb

while True:
    answer = cb.predict(input())
    print(answer)

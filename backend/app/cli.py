from chatbot import processPDF,answer,getSerializedVectorStore

msgs = []

chunks = processPDF('./uploads/const.pdf')
vs = getSerializedVectorStore(chunks)

print('Ready!')

while True:
  user_inp = input('>')
  ans = answer(vs, user_inp, msgs)
  print('Standalone Q:')
  print(ans.get('standalone_q'))
  print('Context:')
  print(ans.get('context'))
  print('Answer:')
  print(ans.get('answer'))
  msgs.append(user_inp)
  msgs.append(ans.get('answer'))

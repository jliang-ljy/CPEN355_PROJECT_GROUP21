import pickle

# 加载并查看 model.p 的内容
with open('./modelknn.p', 'rb') as file:
    model_dict = pickle.load(file)

print(model_dict)

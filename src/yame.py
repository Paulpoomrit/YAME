import pickle

from scripts.feature_extractor import extract_features


with open('model/yame.pkl', 'rb') as f:
    model = pickle.load(f)

while True:
    print('\n⭐🌙⭐🌙⭐🌙⭐------------Yame: AI detector model------------⭐🌙⭐🌙⭐🌙⭐\n')
    print("type 'slop' to exit the program\n")
    input_str = input("Enter text: ")

    if input_str.lower() == 'slop':
        break

    input_vec = extract_features(input_str).reshape(1, -1)

    print('\n----------------')
    print(model.predict(input_vec))
    print('----------------')
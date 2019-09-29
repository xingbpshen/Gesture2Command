import numpy as np
import joblib
import json
import os
import time

model = joblib.load('awesome.pkl')
dict = {0: "this is command 0",
        1: "this is command 1",
        2: "this is command 2",
        3: "this is command 3",
        4: "this is command 4",
        5: "this is command 5",
        6: "this is command 6",
        7: "this is command 7"}


def load_data(file_name):
    while True:
        if os.path.exists(file_name):
            with open(file_name, 'r') as open_file:
                data = open_file.read()
            val = json.loads(data)
            arr = val["0"]
            arr = np.array(arr)
            if len(arr) == 15:
                arr = np.reshape(arr, (15, 28))
                return arr
            else:
                # print("sb")
                time.sleep(1)
        else:
            time.sleep(1)


test_X = load_data("temp.json")
a = np.array([0, 0, 0, 0, 0, 0, 0, 0])
for i in range(0, 15):
    est = model.predict(np.reshape(test_X[i], (1, 28)))
    est_pos = np.argmax(est)
    a[est_pos] += 1

print("\n\n\n\n")
print(dict[np.argmax(a)])
os.system("del temp.json")
input("HAHAHA")

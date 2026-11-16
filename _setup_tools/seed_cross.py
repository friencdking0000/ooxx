import json, os
OUT = r"C:\Users\GIGABYTE\Desktop\OOXX-robot-v1.1\Arena\out"
json.dump(dict(time="18:41", img=os.path.join(OUT, "place_20261115_184410.jpg"),
               crosses={"c0": [588.0, 292.5], "c1": [706.0, 292.5], "c2": [588.0, 441.0], "c3": [706.0, 441.0]}),
          open(os.path.join(OUT, "crosses_last.json"), "w", encoding="utf-8"))
print("seeded")

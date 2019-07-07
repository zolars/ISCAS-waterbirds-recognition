import io

origin = open("classes.txt", "r").readlines()

en = open("enhence_classes.txt", "r").readlines()

enName = []
for i in en:
    enName.append(i.split(" ")[1])

originChange = []
for a in origin:
    if a in enName:
        originChange.append(en[enName.index(a)])
    else:
        originChange.append("ERR "+a)

for a in originChange:
    print(a)

import os
pth = "C:\\Users\\rlaal\\OneDrive\\Desktop\\Training\\noisy"

lst = os.listdir(pth)

result = []

for i in lst:
    exten = i[i.index('.')-2:i.index('.')]
    if exten not in result:
        result.append(exten)
print(result)
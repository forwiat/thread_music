dic = {'haha':0, 'h2':1}
labels = ['haha', 'h2']
a = [0,0,0,0,0]
res = [dic[i] for i in labels]
for i in res:
    a[i] = 1
print(a)

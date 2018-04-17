f1 = open('output0.txt').readlines()
f2 = open('output1.txt').readlines()
f3 = open('output2.txt').readlines()
f4 = open('output3.txt').readlines()
f5 = open('output4.txt').readlines()
f6 = open('output_combine.txt').readlines()

for i in range(len(f1)):
    if f1[i].startswith('-DOCSTART-'):
        continue
    if f1[i] == '\n':
        print('\n')
        continue
    word, tag1 = f1[i].strip().split(' ')
    _, tag2 = f2[i].strip().split(' ')
    _, tag3 = f3[i].strip().split(' ')
    _, tag4 = f4[i].strip().split(' ')
    _, tag5 = f5[i].strip().split(' ')
    _, tag6 = f6[i].strip().split(' ')
    print('\t'.join([word, tag1, tag2, tag3, tag4, tag5, tag6]))

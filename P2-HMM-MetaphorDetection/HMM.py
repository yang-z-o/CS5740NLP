import csv
import ast
import numpy as np

# read training data
word, pos, md = [], [], []
with open('./data_release/train.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        word.append(line[0].split())
        pos.append(ast.literal_eval(line[1]))
        md.append(ast.literal_eval(line[2]))

# read validation data
val_word, val_pos, val_md = [], [], []
with open('./data_release/val.csv', encoding='latin-1') as f:
    lines = csv.reader(f)
    next(lines)
    for line in lines:
        val_word.append(line[0].split())
        val_pos.append(ast.literal_eval(line[1]))
        val_md.append(ast.literal_eval(line[2]))
    
# emission probability (unk rate = 0.01%)
unigram, md_count = {}, {}
for i in range(len(word)):
    for j in range(len(word[i])):
        unigram[word[i][j]] = unigram.get(word[i][j], 0) + 1
        if md[i][j]:
            md_count[word[i][j]] = md_count.get(word[i][j], 0) + 1

# transition probability
count_s0, count_s1 = 0, 0
count_00, count_01 = 0, 0
count_10, count_11 = 0, 0
for i in range(len(word)):
    for j in range(len(word[i])):
        if j == 0:
            if md[i][0] == 1:
                count_s1 += 1
            else:
                count_s0 += 1
        else:
            if md[i][j-1] == 1:
                if md[i][j] == 1:
                    count_11 += 1
                else:
                    count_10 += 1
            else:
                if md[i][j] == 1:
                    count_01 += 1
                else:
                    count_00 += 1
ps0 = count_s0 / (count_s0 + count_s1)
ps1 = 1 - ps0
p00 = count_00 / (count_00 + count_01)
p01 = 1 - p00
p10 = count_10 / (count_10 + count_11)
p11 = 1 - p10

# predict on validation data 
pred_md = []
for i in range(len(val_word)):
    n = len(val_word[i])
    SCORE = [[0] * n for t in range(2)]
    BPTR = [[0] * n for t in range(2)]
    T = [0] * n
    emission_p = md_count.get(val_word[i][0], 0) / unigram.get(val_word[i][0], 1)
    SCORE[0][0] = ps0 * (1 - emission_p)
    SCORE[1][0] = ps1 * emission_p
    for j in range(1,n):
        emission_p = md_count.get(val_word[i][j], 0) / unigram.get(val_word[i][j], 1)
        t00 = SCORE[0][j-1] * p00 * (1 - emission_p)
        t10 = SCORE[1][j-1] * p10 * (1 - emission_p)
        t01 = SCORE[0][j-1] * p01 * emission_p
        t11 = SCORE[1][j-1] * p11 * emission_p

        if t00 >= t10:
            SCORE[0][j] = t00
            BPTR[0][j] = 0
        else:
            SCORE[0][j] = t10
            BPTR[0][j] = 1
        
        if t01 >= t11:
            SCORE[1][j] = t01
            BPTR[1][j] = 0
        else:
            SCORE[1][j] = t11
            BPTR[1][j] = 1
    if SCORE[0][n-1] <= SCORE[1][n-1]:
        T[n-1] = 1
    for j in range(n-2, -1, -1):
        T[j] = BPTR[int(T[j+1])][j+1]
    pred_md.append(T)

# write output.csv
with open('val_output.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')
    filewriter.writerow(['idx','label'])
    cnt = 0
    for i in range(len(pred_md)):
        for j in range(len(pred_md[i])):
            cnt += 1
            filewriter.writerow([cnt, pred_md[i][j]])
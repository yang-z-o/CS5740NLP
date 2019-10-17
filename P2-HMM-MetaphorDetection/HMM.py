import csv
import ast
import numpy as np
import sys

def add_count(dict,key):
    dict[key] = dict.get(key,0) + 1

def main():
    # read training data
    word, pos, md = [], [], []
    with open('./data_release/train.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            word.append(line[0].split())
            pos.append(ast.literal_eval(line[1]))
            md.append(ast.literal_eval(line[2]))

    # select unknown words
    bow = {}
    unk_num = int(sys.argv[2])
    unk = set()
    for i in range(len(word)):
        for j in range(len(word[i])):
            bow[word[i][j]] = bow.get(word[i][j], 0) + 1
    words = [(w,c) for w,c in bow.items()]
    sorted_words = sorted(words, key=lambda k:k[1])
    for i in range(unk_num):
        unk.add(sorted_words[i][0])   

    # emission probability (unk rate = 0.01%)
    unigram, md0_count, md1_count = {}, {}, {}
    m_pos = set()
    all_pos = set()
    cnt_0, cnt_1 = 0, 0
    for i in range(len(word)):
        for j in range(len(word[i])):
            key = word[i][j]

            # unkown words handling
            if word[i][j] in unk:
                key = 'unk'
            all_pos.add(pos[i][j])
            add_count(unigram, key)
            if md[i][j] is 1:
                m_pos.add(pos[i][j])
                cnt_1 += 1
                add_count(md1_count, key)
            else:
                cnt_0 += 1
                add_count(md0_count, key)
    # print(m_pos, len(m_pos))
    # print(all_pos, len(all_pos))
    # print(all_pos - m_pos)

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
    p00 = count_00 / cnt_0
    p01 = 1 - p00
    p10 = count_10 / cnt_1
    p11 = 1 - p10


    # read validation data
    val_word, val_pos, val_md = [], [], []
    with open('./data_release/test.csv', encoding='latin-1') as f:
        lines = csv.reader(f)
        next(lines)
        for line in lines:
            val_word.append(line[0].split())
            val_pos.append(ast.literal_eval(line[1]))
            #val_md.append(ast.literal_eval(line[2]))

    # predict on validation data 
    #k = float(sys.argv[1])
    #kv = k * len(unigram)
    pred_md = []
    l = float(sys.argv[1])
    # hmm
    emission_unk_0 = md0_count['unk']/ cnt_0
    emission_unk_1 = (md1_count['unk']/ cnt_1) * l

    # logistic regression
    # emission_unk_0 = md0_count['unk']/ unigram['unk']
    # emission_unk_1 = (md1_count['unk'] / unigram['unk']) + l

    for i in range(len(val_word)):

        n = len(val_word[i])
        SCORE = [[0] * n for t in range(2)]
        BPTR = [[0] * n for t in range(2)]
        T = [0] * n
        
        # initailization 
        if unigram.get(val_word[i][0], 0) is 0:
            emission_p0 = emission_unk_0
            emission_p1 = emission_unk_1
        else:
            # hmm
            emission_p0 = md0_count.get(val_word[i][0], 0) / cnt_0 
            emission_p1 = (md1_count.get(val_word[i][0], 0) / cnt_1) * l

            # logistic regression
            # emission_p0 = md0_count.get(val_word[i][0], 0) / unigram[val_word[i][0]]
            # emission_p1 = (md1_count.get(val_word[i][0], 0) / unigram[val_word[i][0]]) + l

        SCORE[0][0] = ps0 * emission_p0
        SCORE[1][0] = ps1 * emission_p1

        # iteration
        for j in range(1,n):
            if unigram.get(val_word[i][j], 0) is 0:
                emission_p0 = emission_unk_0
                emission_p1 = emission_unk_1
            else:
                # hmm
                emission_p0 = md0_count.get(val_word[i][j], 0) / cnt_0
                emission_p1 = (md1_count.get(val_word[i][j], 0) / cnt_1) * l

                # logistic regression
                # emission_p0 = md0_count.get(val_word[i][j], 0) / unigram[val_word[i][j]]
                # emission_p1 = (md1_count.get(val_word[i][j], 0) / unigram[val_word[i][j]]) + l

            t00 = SCORE[0][j-1] * p00 * emission_p0
            t10 = SCORE[1][j-1] * p10 * emission_p0
            t01 = SCORE[0][j-1] * p01 * emission_p1
            t11 = SCORE[1][j-1] * p11 * emission_p1

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
        
        # back forward
        if SCORE[0][n-1] <= SCORE[1][n-1]:
            T[n-1] = 1
        for j in range(n-2, -1, -1):
            T[j] = BPTR[int(T[j+1])][j+1]
        pred_md.append(T)

    # write output.csv
    with open('output.csv', 'w') as csvfile:
        filewriter = csv.writer(csvfile, delimiter=',')
        filewriter.writerow(['idx','label'])
        cnt = 0
        for i in range(len(pred_md)):
            for j in range(len(pred_md[i])):
                cnt += 1
                filewriter.writerow([cnt, pred_md[i][j]])

if __name__ == '__main__':
	main()
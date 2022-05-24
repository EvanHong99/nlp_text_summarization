import jieba
import os

root="./mini_lcsts"
save_root="./prep_data"
jieba.enable_paddle()

for root, dirs, files in os.walk(root):
    print(files)
    for file in files:
        if file[-3:] in ['tgt','src']:
            print(root, files)
            with open(root + '/'+ file , "r", encoding='utf-8') as fr:
                res=[]
                for line in fr.readlines():
                    seg_list = jieba.cut(line.strip(), use_paddle=True)
                    res.append(" ".join(seg_list) + '\n')  
                with open(save_root+ '/' + file, 'w', encoding='utf-8') as fw:
                    fw.writelines(res)

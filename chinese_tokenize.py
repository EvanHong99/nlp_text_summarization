import jieba
import os

root="./mini_lcsts"
save_root="./prep_data"

for root, dirs, files in os.walk(root):
    print(root)
    print(dirs)
    print(files)
    for file in files:
        if file[-3:] in ['tgt','src']:
            print(root + '/' + file)
            with open(root + '/'+ file , "r", encoding='utf-8') as fr:
                res=[]
                for line in fr.readlines():
                    seg_list = jieba.cut(line.strip())
                    res.append(" ".join(seg_list) + '\n')  
                with open(save_root+ '/' + file, 'w', encoding='utf-8') as fw:
                    fw.writelines(res)

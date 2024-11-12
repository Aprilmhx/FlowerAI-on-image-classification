import os
import shutil

file_path1 = "isic2019_all"
file_path2 = "isic2019"

read_csv = open("isic2019/ISIC_2019_Training_GroundTruth.csv").read().splitlines()
file_dict = {}
for line in read_csv[1:]:
    filename,MEL,NV,BCC,AK,BKL,DF,VASC,SCC,UNK = line.strip().split(",")    
    if MEL == str(1.0):
        file_dict[filename.strip()] = str(1)
    elif NV == str(1.0):
        file_dict[filename.strip()] = str(2)
    elif BCC == str(1.0):
        file_dict[filename.strip()] = str(3)
    elif AK == str(1.0):
        file_dict[filename.strip()] = str(4)   
    elif BKL == str(1.0):
        file_dict[filename.strip()] = str(5)  
    elif DF == str(1.0):
        file_dict[filename.strip()] = str(6)    
    elif VASC == str(1.0):
        file_dict[filename.strip()] = str(7)
    elif SCC == str(1.0):
        file_dict[filename.strip()] = str(8)    
    else:        
        file_dict[filename.strip()] = str(9)
        
filenames = [s.split('.')[0] for s in os.listdir(file_path1) if '.jpg' in s]  

for f in filenames: 
    shutil.copy(os.path.join(file_path1,f+".jpg"), os.path.join(file_path2,file_dict[f]))     


import os
import csv
import shutil
def load(path):
    images = []
    with open(path+"/imageLists.csv", "w", newline='') as f:
        wr = csv.writer(f)
        for label in os.listdir(path) :
            for file in os.listdir(path+'/'+label):
                name, _, ext = file.partition('.')
                labels = [str(i) for i in label.split('-')]
                labels= (',').join(labels)
                if( ext.lower() in ['png', 'jpg', 'jpeg', 'bmp', 'psd'] ):
                    wr.writerow([name+'.'+ext, labels])

def read_all_file(path):
    output = os.listdir(path)
    file_list = []
    for i in output:
        if os.path.isdir(path+"/"+i):
            file_list.extend(read_all_file(path+"/"+i))
        elif os.path.isfile(path+"/"+i):
            file_list.append(path+"/"+i)
    return file_list


def copy_all_file(file_list, new_path):
    for src_path in file_list:
        file = src_path.split("/")[-1]
        shutil.copyfile(src_path, new_path+"/"+file)

src_path='../dataset/radiograph_excel/test'
new_path='../dataset/radiograph_excel/test/test'
load(src_path)
# file_list = read_all_file(src_path)
# copy_all_file(file_list, new_path)

# xk97
import os
import subprocess
import gzip
import shutil
# import fnmatch

def walk_folder():
    for root, _, files in os.walk("."):
        #         print('root-----------', root, '\n')
        #         for items in fnmatch.filter(files, "*"):
        #                 print ("..." + items)
        for items in os.listdir(root):
            #             print(items)
            if (items[:1] == '.'):
                #                 os.remove(os.path.join(root, items))
                print(os.path.join(root, items))
    #                 print("File Removed!", items, '\n')

def unzip():
    for file_ in os.listdir('.'):
        if not '.gz' in file_:
            continue
        with gzip.open(file_, 'rb') as fin, \
                open(file_.replace('.gz', ''), 'wb') as fout:
            print('uzipping', file_)
            shutil.copyfileobj(fin, fout)

def run_exe(*args):
    pr = subprocess.run(['ls', '../'], capture_output=False)
    print(pr.stdout)

    p = subprocess.Popen(['notepad', './test.py'], stdin=None, stdout=subprocess.PIPE)
    if p.stdout.readlines(): 
        [print(_) for _ in p.stdout.readlines()]

if __name__ == "__main__":
    import sys
    argv = sys.argv[1:] or ['run_exe']
    print(argv)
    if 'run_exe' in argv:
        run_exe(*argv[1:])
    if 'walk_folder' in argv:
        run_exe(*argv[1:])
    if 'unzip' in argv:
        run_exe(*argv[1:])
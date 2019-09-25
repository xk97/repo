# xk97
import os
# import fnmatch

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

import sys
import os

gitignore_text= '''
LOG/*
*.png
*.hdf5
'''
assert(len(sys.argv) >=2)

tut_numb = int(sys.argv[1])
tut_name = " ".join(sys.argv[2:])


tut_dir = "Tut{0:06}".format(tut_numb)
os.mkdir(tut_dir)
os.mkdir(tut_dir+"/LOG")
f= open(tut_dir+"/.gitignore","w")
f.write(gitignore_text)
f.close()

f= open("log.txt","a")
f.write("Tut{0:06}:  {1}\n".format(tut_numb,tut_name))
f.close()

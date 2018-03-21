import ftplib
from time import sleep
import os
import pickle

"This file recursively searches the ifremer DAC for traj files and makes a list for download. After I wrote this, I learned that there is a list maintained on the server of this exact thing, so it is essentially a waste. I will save this for posterity in case I need a similar script again"

class argo_traj_data:
	def __init__(self):
		ftp_link = 'ftp.ifremer.fr'
		self.ftp = ftplib.FTP(ftp_link)
		self.ftp.login()
		self. my_dirs = []
		self.my_files = []
		self.check_dir('/ifremer/argo/dac')

	def get_dirs(self,ln):
		cols = ln.split(' ')
		objname = cols[len(cols)-1] # file or directory name
		if ln.startswith('d'):
			if objname!='profiles':	#from the data structure there is never a traj file in this directory
				self.my_dirs.append(objname)
		else:
			if objname.endswith('Rtraj.nc'):
				self.my_files.append(os.path.join(self.curdir, objname)) # full path
			if len(self.my_files)%100==0:
				self.save()

	def check_dir(self,adir):
	  self.my_dirs = []
	  gotdirs = [] # local
	  self.curdir = self.ftp.pwd()
	  print("going to change to directory " + adir + " from " + self.curdir)
	  self.ftp.cwd(adir)
	  self.curdir = self.ftp.pwd()
	  print("now in directory: " + self.curdir)
	  self.ftp.retrlines('LIST', self.get_dirs)
	  gotdirs = self.my_dirs
	  print("found in " + adir + " directories:")
	  print(gotdirs)
	  print("Total files found so far: " + str(len(self.my_files)) + ".")
	  for subdir in gotdirs:
	    self.my_dirs = []
	    self.check_dir(subdir) # recurse  
	  self.ftp.cwd('..') # back up a directory when done here

	def save(self):
		with open("backup.txt", "wb") as fp:   #Pickling
			pickle.dump(self.my_files, fp)


a = argo_traj_data()
a.save()
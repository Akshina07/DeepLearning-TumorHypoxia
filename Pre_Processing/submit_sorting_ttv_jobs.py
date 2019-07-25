import os
if __name__=="__main__":
	for i in range(262): #number of directories
		outpath="/gpfs/ysm/scratch60/aj557/tiles/tile_out{}".format(i+1)
		if not os.path.exists(outpath):
			#print("Yes")
			continue 
		else:
			if not os.path.exists(outpath+"/All"):
				#print(i+1)
				#os.system("conda activate openslide-python")
				os.system("python sort_train_test_valid.py --SourceFolder={} --JsonFile=\"/ysm-gpfs/pi/gerstein/aj557/data_deeppath/files.2019-05-31.json\" --Magnification=20  --MagDiffAllowed=15 --PercentTest=15 --PercentValid=15 --PatientID=12 ".format(outpath))
			else:
				continue

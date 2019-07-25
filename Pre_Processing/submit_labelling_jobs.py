import os
for i in range(262): #number of directories
	if not os.path.exists("/ysm-gpfs/pi/gerstein/aj557/data_deeppath/tiles_done/dir_{}".format(i+1)):
		continue
	os.system("python label_images.py --SourceFolder=/ysm-gpfs/pi/gerstein/aj557/data_deeppath/images_input/images{}/".format(i+1))
	print("Done {}".format(i+1))


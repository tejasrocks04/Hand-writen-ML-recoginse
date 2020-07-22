import pyscreenshot as ImageGrab
import time,cv2,joblib

images_folder = "Images/"

	


for i in range (0,45):
	print("draw letter now")

	time.sleep(2)
	im = ImageGrab.grab(bbox=(110,182 , 344, 381)) # X1,Y1,X2,Y2
	print ("saved....",i)
	im.save(images_folder+str(i)+'.png')
	#im.save(str(i)+".png")
	print("clear now")
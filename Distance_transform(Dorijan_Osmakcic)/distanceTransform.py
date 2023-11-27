import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import cv2 as cv
from sklearn.metrics import mean_squared_error
from scipy import ndimage
from skimage import data
from skimage.util import invert
import time





def get_dst_transform_img(og): #og is a numpy array of original image
   cords=[]
   zeros_loc = np.where(og == 0 )
   zeros = np.asarray(zeros_loc).T # coords of all zeros in og

   for i in range(zeros.shape[0]):
      if(zeros[i,0]-1>=0 and zeros[i,1]-1>=0 and zeros[i,0]+1<=og.shape[0]-1 and zeros[i,1]+1<=og.shape[1]-1):
        if(og[zeros[i,0],zeros[i,1]-1]==1 or og[zeros[i,0]-1,zeros[i,1]-1]==1 or og[zeros[i,0]-1,zeros[i,1]]==1 or 
        og[zeros[i,0]+1,zeros[i,1]-1]==1 or og[zeros[i,0]-1,zeros[i,1]+1]==1 or og[zeros[i,0],zeros[i,1]+1]==1 or 
        og[zeros[i,0]+1,zeros[i,1]]==1 or og[zeros[i,0]+1,zeros[i,1]+1]==1):
            cords.append([zeros[i,0],zeros[i,1]])


  
   ones_loc = np.where(og == 1)
   ones = np.asarray(ones_loc).T # coords of all ones in og

   a = -2 * np.dot(ones, np.asarray(cords).T) 
   b = np.sum(np.square(cords), axis=1) 
   c = np.sum(np.square(ones), axis=1)[:,np.newaxis]
   dists = a + b + c
   dists = np.sqrt(dists.min(axis=1)) # min dist of each zero pixel to one pixel
   x = og.shape[0]
   y = og.shape[1]
   dist_transform = np.ones((x,y))
   dist_transform[ones[:,0], ones[:,1]] = dists

   return dist_transform








img=cv.imread(r'C:\Users\dorij\OneDrive\Desktop\Nova mapa\brain.jpg')
x=np.arange(0.0,1.0,0.1)
polje=[]



for i in  (0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1):
   img_75 = cv.resize(img, None, fx =i, fy =i)
   gray = cv.cvtColor(img_75, cv.COLOR_BGR2GRAY)

   grayreal=np.where(gray>200,1,0)

   threshdata=np.asarray(grayreal,dtype='uint8')

   start1=time.time()*1000
   arr=ndimage.distance_transform_edt(threshdata)
   end1=time.time()*1000
   print((end1-start1),'ms of Scipy')
   



   start2=time.time()*1000
   dstmy=get_dst_transform_img(threshdata)
   end2=time.time()*1000
   print((end2-start2),'ms of my DistTransform')
   


   start3=time.time()*1000
   distcv = cv.distanceTransform(threshdata, cv.DIST_L2, 5)
   end3=time.time()*1000
   print((end3-start3),'ms of OpenCV')
   
   polje.append([(end1-start1),(end2-start2),(end3-start3)])

  





print(polje)

ls=np.array(polje)

plt.figure(6)
plt.imshow(threshdata,cmap='gray')
plt.show()



plt.plot(x, ls[:,0], label = "Scipy")
plt.plot(x,ls[:,1] , label = "With numpy")
plt.plot(x, ls[:,2], label = "OpenCV")
plt.xlabel('Scale  of picture ')
plt.ylabel('Time [ms]')

plt.legend(["Scikit distance transform","My distance transform","OpenCV distance transform"])
plt.show()



f, axarr = plt.subplots(1,3)
axarr[0].imshow(arr,cmap = 'gray')
axarr[0].set_title('SCIKIT')
axarr[1].imshow(dstmy,cmap = 'gray')
axarr[1].set_title('USER FUNCTION')
axarr[2].imshow(distcv,cmap = 'gray')
axarr[2].set_title('OPENCV')

plt.show()











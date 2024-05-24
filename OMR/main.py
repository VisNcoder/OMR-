import cv2
import numpy as np
import utilis

path = "l.png"
widthImg=500
heightImg=500
rectcont=[]
img=cv2.imread(path)

ans=[1,1,1,2,3,1,1,1,2,3,1,1,1,2,3,1,1,1,2,3]

img=cv2.resize(img, (widthImg, heightImg))
imgbiggestcontour=img.copy()
imgcontours=img.copy()
imgWarpcolored=img.copy()
imgGray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray, (5,5),1)
imgCanny=cv2.Canny(imgBlur,10,50)


countours , hierarchy=cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgcontours, countours,-1, (0,255,0),10)


rectcont=utilis.rectcontour(countours)
big1=utilis.getcornerpoint(rectcont[0])
# big2=utilis.getcornerpoint(rectcont[1])


if big1.size != 0:

    cv2.drawContours(imgbiggestcontour,big1,-1,(0,255,0),20)
    # cv2.drawContours(imgbiggestcontour,big2,-1,(255,0,0),20)
   

   
    big1=utilis.reorderpoints(big1)
    # big2=utilis.reorderpoints(big2)
  
   
    pts1=np.float32(big1)
    pts2=np.float32([[0,0], [widthImg,0] ,[0, heightImg] , [widthImg, heightImg]])
    matrix=cv2.getPerspectiveTransform(pts1, pts2)
    imgWarpcolored=cv2.warpPerspective(img, matrix, (widthImg, heightImg))

    # pts3=np.float32(big2)
    # pts4=np.float32([[0,0], [widthImg,0] ,[0, heightImg] , [widthImg, heightImg]])
    # matrix1=cv2.getPerspectiveTransform(pts3, pts4)
    # imgWarpcolored=cv2.warpPerspective(imgWarpcolored, matrix1, (widthImg, heightImg))


    imgWarpGray = cv2.cvtColor(imgWarpcolored,cv2.COLOR_BGR2GRAY) # CONVERT TO GRAYSCALE
    imgThresh = cv2.threshold(imgWarpGray, 170, 255,cv2.THRESH_BINARY_INV )[1] # APPLY THRESHOLD AND INVERSE


    boxes=utilis.splitBoxes(imgThresh)
    
    mypixelval=np.zeros((20,4))
    countr=0
    countc=0

    for image in boxes:
        totalpixel=cv2.countNonZero(image)
        mypixelval[countr][countc]=totalpixel
        countc+=1
        if(countc==4):
            countr+=1
            countc=0
    
    myindex=[]

    for x in range (0, 20):
        arr=mypixelval[x]
        myIndexval=np.where(arr==np.amax(arr))
        myindex.append(myIndexval[0][0])
    print(myindex)

    grading=[]

    for x in range(0, 20):
        if ans[x]==myindex[x]:
            grading.append(1)
        else:
            grading.append(0)    
    print(grading)

    score=(sum(grading)/20) * 100
    print(score)



    

imgBlank=np.zeros_like(img)

imageArray=([img, imgGray,imgBlur,imgCanny], [imgcontours,imgbiggestcontour,imgWarpcolored,imgThresh])
imgStacked=utilis.stackImages(imageArray, 0.5)



cv2.imshow("Original", imgStacked)
cv2.waitKey(0)
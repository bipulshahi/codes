img=cv2.imread(r"C:/Users/sd/Desktop/1.png",0)
cv2.imshow("image",img)
k=cv2.waitKey(0)
if k==27:
    cv2.destroyAllWindows()
elif k== ord('s'):
    cv2.imwrite(r"C:\Users\Shubh_Ram\Desktop\b.jpg",img)
    cv2.destroyAllWindows()
    

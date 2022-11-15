import numpy as np
import cv2 as cv 
import matplotlib.pyplot as plt
from sift import Sift

bf = cv.BFMatcher(cv.NORM_L1, crossCheck=True)


class MyImage:
    def __init__(self,image,text):
        self.image=image
        self.text=text

text = """Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever
since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries,
but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets
containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum."""

def char_to_binary(a):
    if type(a) == str:
        return char_to_binary(char_to_ascii(a))
    ans = char_to_binary_aux(a)
    if(len(ans)<8):
        ans=(8-len(ans))*"0"+ans
    return ans        

def char_to_binary_aux(a):
    ans=""
    if a >= 1 :
         ans = char_to_binary_aux(a//2)+str(a%2)
    else: return ""
    return ans
def char_to_ascii(c):
    return ord(c)

def string_to_binary(s):
    ans=""
    for c in s:
        ans+= char_to_binary(ord(c))
    return ans


def binary_to_char(binary):
    ans = 0
    for i in range(len(binary)):
        ans += int(binary[i])*(2**(len(binary)-i-1))
    return ans
    
def encrypt(s,arr):
    encrypted = string_to_binary(s)
    count = 0
    for i,j in enumerate(arr):
        if i > len(encrypted)-1: break
        binary = char_to_binary(j)
        new_bin = binary[:-1] + encrypted[count]
        count+=1
        arr[i] = binary_to_char(new_bin)
    ending_op = string_to_binary("*")
    for i,j in enumerate(range(count,count+8)):
        binary = char_to_binary(arr[j])
        new_bin = binary[:-1] + ending_op[i]
        arr[j] = binary_to_char(new_bin)
    return arr
def ascii_to_char(a):
    return chr(a)

def decrypt(arr):
    decrypted = ""
    for i in range(0,len(arr),8):
        temp = ""
        for j in range(8):
            curr = char_to_binary(arr[i+j])
            temp += curr[-1]
        temp = binary_to_char(temp)
        if ascii_to_char(temp) == "*": break
        decrypted+=ascii_to_char(temp)
    return decrypted

def encrypt_image_inside_image(image,secret):
    if image.shape[0] != secret.shape[0] or image.shape[1] != secret.shape[1]:
        print("Image and secret must be the same size")
        return
    output = np.zeros((image.shape[0],image.shape[1]))
    for i in range(secret.shape[0]):
        for j in range(secret.shape[1]):
            binary1 = char_to_binary(image[i,j])
            binary2 = char_to_binary(secret[i,j])
            output[i,j] = binary_to_char(binary1[:5]+binary2[:3])
    return output.astype("uint8")        

def decrypt_image_inside_image(image):
    output = np.zeros((image.shape[0],image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            binary = char_to_binary(image[i,j])
            output[i,j] = binary_to_char(binary[5:]+"0"*5)
    return output.astype("uint8")

def calculateMatches(descriptor1,descriptor2):
    matches = bf.match(descriptor1,descriptor2)
    matches = sorted(matches, key = lambda x:x.distance)
    return matches        

secret = cv.imread("house512.jpg",0)
img = cv.imread("3.jpg",0)
img2 = cv.imread("face3.png",0)

encrypted_image = encrypt_image_inside_image(img,secret)

decrypted_image = decrypt_image_inside_image(encrypted_image)




encrypted = encrypt(text,img.ravel()).reshape((img.shape[0],img.shape[1]))
message = decrypt(encrypted.ravel())

cv.imshow("secret",secret)
cv.imshow("original image",img)
cv.imshow("encrypted image",encrypted)
cv.imshow("foto-key",img2)
cv.imshow("decrypted_image",decrypted_image)
cv.waitKey(0)
cv.destroyAllWindows()
print(message)


mysift = Sift((2**(0.5)),((2**(0.5))/2),4,5)
sift = cv.SIFT_create()

#keypoint1,descriptor1 = sift.detectAndCompute(decrypted_image, None)
#keypoint2,descriptor2 = sift.detectAndCompute(img2, None)
#matches = calculateMatches(descriptor1,descriptor2)
#matchPlot = cv.drawMatches(img, keypoint1, img2, keypoint2, matches[:10], img2, flags=2)
#plt.imshow(matchPlot)

keypoint1,descriptor1 = mysift.detectAndCompute(decrypted_image)
keypoint2,descriptor2 = mysift.detectAndCompute(img2)
matches = calculateMatches(descriptor1,descriptor2)
matchPlot3 = cv.drawMatches(img, keypoint1, img2, keypoint2, matches[:10], img2, flags=2)
plt.imshow(matchPlot3)

#matchPlot3 = cv.drawKeypoints(img, keypoint5, None, flags=0)
#matchPlot3 = cv.drawKeypoints(img, keypoint5, None, flags=0)
plt.show()

# <p align="center">Steganograpy with SIFT Project for Image Process Class </p> </br>

## <p align="center">You can review the presentation (huseyin_pekkan_201401016_project_report.docx) that I prepared for this problem in the Image Process Course. The presentation is in Turkish.</p> 

## <p align="center">English description is available below</p> 

### <p align="center">My aim was to combine steganography and sift, using a key to extract the area that was wanted to be shown to us in photos shared in public. For example, a government official deciphers the photograph that is publicly broadcast on television and determines the point that is intended to be shown in the photograph.</p>



### <p align="center">First, let's assume that we have an encrypted photo, we convert the value in each pixel of this photo to an 8-bit binary number and take the last 3 bits and add 5 "0"s to it to get our hidden photo. We assume that the person who wants to do this has a photo key in his hand. We SIFT the obscured photo and the photo key, and the matching points give us the points we need to focus on in the main photo. Here, it is important that the main photo and the hidden photo are the same sizes.</p>
### <p align="center">Decryption</p>
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201935184-e9c936ce-cd43-4d3a-93fb-a4ca57d3cb46.png" alt="">
</p></br>

### <p align="center">Encryption</p>
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201935523-34666693-dcb8-4bf3-8048-a74be2258f35.png" alt="">
</p></br> 
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201935890-517c062d-dcd9-4bb3-b881-8029d54c102b.png" alt="">
</p></br> 

### <p align="center">It takes k, sigma, octave, and scale variables in order and creates a SIFT object. The k value here is √2 and the sigma value is (√2)/2 in the article. The octave and scale values can be calculated from these values, but I didn't bother to calculate these values because they were calculated in the article. I also create SIFT of openCV to compare.</p>
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201936311-f78e7118-9034-44ed-9650-863ecea5d94f.png" alt="">
</p></br> 

### <p align="center">Here, we extract the SIFT of the photo we decrypted and the SIFT of the photo-key img2. And we find the matches with the help of descriptors, but we show these matches on the original photo, img, so we determine the region that is wanted to be shown on the original photo.
</p>

<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201936608-50d0e38a-1b3b-4001-9812-f7b2692295f4.png" alt="">
</p></br> 

### <p align="center">It creates Scale space and DoG space. It extracts keypoints from maximum and minimum points using DoG space. Eliminates non-unique keypoints, resizes them. And it extracts descriptors from keypoints. And lastly , returns the keypoints as the myKeyPoint class.</p>

<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201937374-36a6b627-340e-452a-bd5a-659539459d0b.png" alt="">
</p></br> 

### <p align="center">It creates a gaussian image space with the size of "octave x scale" from the photo it takes. The upsampling we did here at the beginning allows us to get more keypoints as there is more detail. By applying the sigma changes in the article, it blurs the photo by reducing it 2x each time.
</p>

<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201937947-d18817e5-5f4c-4858-b514-dfb43b0461b6.png" alt="">
</p></br> 

### <p align="center">Here our aim is to check whether the center is maximum or minimum using 26 neighbors. Neighbors are dogSpace[i,j-1], dogSpace[i,j+1] and dogSpace[i,j] (not including center) and localization is required as in the main article because if a continuous function is captured then a better local maximum or minimum can be found.</p>

<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201938616-fc5e006a-0db3-4420-98bf-d264dc064043.png" alt="">
</p></br> 

### <p align="center">Here our goal is to find the best local maximum or minimum. While doing this, we make use of the gradient and hessian matrix. We repeat this up to 5 times. If it changes less than 0.5 in all 3 directions, we have found a local maximum or minimum, and we terminate the loop. If we have reached the limit of 5 cycles specified in the article and we still have not found it, we say that the local maximum is not in this window and we return None type. If we find a local extreme and decide that it is not a low contrast point, we check for edge and flatness. We decide whether there are corners and flats by using the hessian matrix we have created. If it is not an edge or flat area, we create the keypoint and rotate it.</p>

<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201939096-59c1740c-667b-4a59-91c3-993d89ced69b.png" alt="">
</p></br> 

### <p align="center">The goal here is to gain independence without rotating. To do this, we simply create a histogram with 10 degrees and 36 parts for 360 degrees. For each degree, we calculate the gradient magnitude using gaussian images. According to the article, the scale should be 1.5 times. And we take the parts with a peak higher than 0.8 and add the orientation to this keypoint and save it. Then we take 16x16 neighbors and divide it into 16 blocks of 4x4. We create an orientation histogram with 8 parts for each part. So we get our descriptor vector with 8x16=128 parts. We then apply trilinear interpolation to smooth out the weighted gradient magnitude. Then we pass it through a threshold and normalize it.
</p>
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201939551-3cb8466f-fca1-4915-8c45-50eb28825bb4.png" alt="">
</p></br> 
# <p align="center">Experiments and Results</p>
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201940066-6a0c75a5-4335-4090-9585-8211cba2260a.png" alt="">
</p></br> 
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201941259-0fb341fe-241d-4540-b7bd-299fcbf7b511.png" alt="">
</p></br> 
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201941553-a378cdf1-da92-4a97-bc35-913a119a7960.png" alt="">
</p></br> 
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201941805-73a3d27a-490d-48d0-b209-0a8ebf03b601.png" alt="">
</p></br> 
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201941988-eb57eafd-1f9b-417d-b25b-cacc52bf5cbe.png" alt="">
</p></br> 
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201942556-bc5c1e74-c8b5-4b6a-8f8f-b5f9cd4cc65f.png" alt="">
</p></br> 
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201942716-14a2ce8e-4ad2-49a8-98a9-80c844bb18c9.png" alt="">
</p></br> 
<p align="center" ><img src="https://user-images.githubusercontent.com/75019129/201942870-15c40fea-b8fb-4476-91e0-35969a17fbc3.png" alt="">
</p></br> 





















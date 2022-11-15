
import numpy as np
import cv2 as cv
from functools import cmp_to_key


MAX_ALLOWED_WIDTH = 720
MIN_ALLOWED_WIDTH = 720

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv.INTER_LINEAR):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv.resize(image, dim, interpolation=inter)

class myKeyPoint:
    def __init__(self,keypoint,scale,layer):
        self.keypoint=keypoint
        self.scale=scale
        self.layer = layer
        
class Gaussian:
        
    def gaussian(K,r,sigma):
        return K*np.exp(-(((r)**2)/(2*(sigma**2))))
    
    def gaussianKernel(K,f_size,sigma):
        kernel = np.zeros(shape=(f_size,1))
        for x in range(f_size):
            r = (x-(f_size-1)/2)
            kernel[x]=Gaussian.gaussian(K=K,r=r,sigma=sigma)
        return kernel/kernel.sum()
    
    def gaussianFilter(img,f_size,sigma,K=1,padding=cv.BORDER_DEFAULT):
        h,w=img.shape[0],img.shape[1]
        output = np.zeros((h,w))
        gaussianKernel = np.round_(Gaussian.gaussianKernel(K=K,f_size=f_size,sigma=sigma),4)
        output = (cv.sepFilter2D(img,-1,kernelX=gaussianKernel,kernelY=gaussianKernel,borderType=padding))
        return output
    
float_tolerance = 1e-7
class Sift:
    def __init__(self,k,sigma,octave,scale):
        self.k = k
        self.sigma = sigma
        self.octave = octave
        self.scale = scale
        
    def detectAndCompute(self,image):
        image = image.astype("float32")
        scaleSpace,dogSpace = self.createDoG(image)
        keypoints = Sift.createMaxMinSub(self,scaleSpace,dogSpace)
        print("maxmin")
        print(len(keypoints))
        keypoints = Sift.eleminateDuplicates(keypoints)
        print("duplicate")
        keypoints = Sift.convertSize(keypoints)
        print("convert")
        descriptors = Sift.generateDescriptors(keypoints,scaleSpace)
        print("descriptor")
        result = []
        for keypoint in keypoints:
            result.append(keypoint.keypoint)
        return result,descriptors
    
    
    def createSpaceArray(octave,scale,width,height):
        tmp = []
        for i in range(octave):
            for j in range(scale):
                tmp.append(np.zeros((width,height))) 
                width = width//2
                height = height//2
        return np.array(tmp,dtype=object).reshape((octave,scale))
    
    def createScaleSpace(self,image):
        image = ResizeWithAspectRatio(image,width=image.shape[0]*2,height=image.shape[1]*2) #upsampling for better results
        
        space = Sift.createSpaceArray(self.octave,self.scale,image.shape[0],image.shape[1])
        sigma = tmpsigma = self.sigma
        
        for i in range(self.octave):
            if i !=0:
                sigma = 2*sigma
                tmpsigma = sigma
            for j in range(self.scale):
                blurred = cv.GaussianBlur(image,(0,0),sigmaX=tmpsigma)#Gaussian.gaussianFilter(img=image,f_size=7,sigma=tmpsigma)
                #cv.imshow("blr"+str(i)+str(j),blurred.astype("uint8"))
                space[i,j]=blurred
                tmpsigma = self.k*tmpsigma
            image = image[0::2,0::2]
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return space
    
    def createDoG(self,image):
        scaleSpace = self.createScaleSpace(image)
        dogSpace = Sift.createSpaceArray(self.octave,self.scale-1,scaleSpace[0,0].shape[0],scaleSpace[0,0].shape[1])
        for i in range(self.octave):
            for j in range(self.scale-1):
                dogSpace[i,j]=np.subtract(scaleSpace[i][j+1],scaleSpace[i][j])
                #cv.imshow(str(i)+str(j),dogSpace[i,j].astype("uint8"))     
        #cv.waitKey(0)
        #cv.destroyAllWindows()
        return scaleSpace,dogSpace
    
    def createMaxMinSub(self,scaleSpace,dogSpace,image_border_width=5):
       
        threshold = np.floor(0.5 * 0.03 / 3 * 255 )
        keypoint=[]
        f_size = 3
        keypoints = []
        for octave in range(self.octave):
            for scale in range(1,self.scale-2):
                
                stacked_img = np.array([dogSpace[octave,scale-1],dogSpace[octave,scale],dogSpace[octave,scale+1]],dtype=float)
                for x in range(dogSpace[octave,scale+1].shape[0]-image_border_width):
                    for y in range(dogSpace[octave,scale+1].shape[1]-image_border_width):
                        window = np.array(stacked_img[:,x:x+f_size,y:y+f_size],dtype=float)
                        if Sift.isExtremum(window,threshold):
                            localization = Sift.localizationQuadratic(self,x,y,octave,scale,dogSpace,image_border_width)
                            if localization is not None:
                                keypoint, scale_index = localization
                                keypoint_oriantations = Sift.keypointsOriantations(self,keypoint,octave,scaleSpace[octave,scale_index])
                                for keypoint_with_ori in keypoint_oriantations:
                                    if keypoint_with_ori is not None:
                                        keypoints.append(keypoint_with_ori)
        return keypoints
    
    
    def eleminateDuplicates(keypoints):
        if len(keypoints) <=1:
            return keypoints
        keypoints.sort(key=cmp_to_key(Sift.compareKeypoints))
        uniques = [keypoints[0]]
        for next_keypoint in keypoints[1:]:
            temp = next_keypoint.keypoint
            last_unique = uniques[-1].keypoint
            if last_unique.pt[0] != temp.pt[0] or last_unique.pt[1] != temp.pt[1] or last_unique.size != temp.size or last_unique.angle != temp.angle:
                uniques.append(next_keypoint)
        return uniques
        
    def compareKeypoints(keypoint1, keypoint2):
        first = keypoint1.keypoint
        second = keypoint2.keypoint
        if first.pt[0] != second.pt[0]:
            return first.pt[0] - second.pt[0]
        if first.pt[1] != second.pt[1]: 
            return first.pt[1] - second.pt[1]
        if first.size != second.size:
            return second.size - first.size
        if first.angle != second.angle:
            return first.angle - second.angle
        if first.response != second.response:
            return second.response - first.response
        if first.octave != second.octave:
            return second.octave - first.octave
        return second.class_id - first.class_id
    
    def keypointsOriantations(self,keypoint,octave,scaleImage):
        keypointsOriantations = []
        shape = scaleImage.shape
        MyKeyPoint = keypoint
        keypoint = keypoint.keypoint
        scale = 1.5 * keypoint.size / np.float32(2 ** (octave + 1))
        radius = int(np.round_(3 * scale))
        weight_factor = -0.5 * (scale ** 2)
        raw_hist = np.zeros(36)
        smooth_hist = np.zeros(36)
        for i in range(-radius,radius+1):
            region_y = int(np.round_(keypoint.pt[1] / np.float32(2 ** octave))) + i
            if region_y > 0 and region_y < shape[0] - 1:
                 for j in range(-radius,radius + 1):
                    region_x = int(np.round_(keypoint.pt[0] / np.float32(2 ** octave))) + j
                    if region_x > 0 and region_x < shape[1] - 1:
                        dx = scaleImage[region_y,region_x+1] - scaleImage[region_y,region_x-1]
                        dy = scaleImage[region_y-1,region_x] - scaleImage[region_y+1,region_x]
                        gradint_magn = np.sqrt(dx*dx+dy*dy)
                        gradient_ori = np.rad2deg(np.arctan2(dy,dx))
                        weight = np.exp(weight_factor * (i ** 2 + j ** 2))
                        histogram_ind = int(np.round_(gradient_ori * 36 / 360.))
                        raw_hist[histogram_ind % 36] += weight * gradint_magn
        for n in range(36):
            smooth_hist[n] = (6 * raw_hist[n] + 4 * (raw_hist[n-1] + raw_hist[(n+1) % 36]) + raw_hist[n-2] + raw_hist[(n+2) % 36]) / 16.
        ori_max = np.max(smooth_hist)
        ori_peaks = np.where(np.logical_and(smooth_hist > np.roll(smooth_hist,1),smooth_hist > np.roll(smooth_hist,-1)))[0]
        for peak_ind in ori_peaks:
            peak_val = smooth_hist[peak_ind]
            if peak_val >= 0.8 * ori_max:
                left_value = smooth_hist[(peak_ind - 1) % 36]
                right_value = smooth_hist[(peak_ind + 1) % 36]
                interpolated_peak_index = (peak_ind + 0.5 * (left_value - right_value) / (left_value - 2 * peak_val + right_value)) % 36
                ori = 360. - interpolated_peak_index * 360. / 36.
                if abs(ori - 360.) < float_tolerance:
                    ori = 0
                new_keypoint = myKeyPoint(cv.KeyPoint(*keypoint.pt,keypoint.size,ori,keypoint.response,keypoint.octave),MyKeyPoint.scale,MyKeyPoint.layer)
                keypointsOriantations.append(new_keypoint)
        return keypointsOriantations
        
        
    def localizationQuadratic(self,x,y,octave,scale,dogSpace,image_border_width):
            isOutside = False
            shape = dogSpace[octave][0].shape
            eigenValue = 10
            limit = 5
            for count in range(limit):
                if x < image_border_width or x >= shape[0] - image_border_width or y < image_border_width or y >= shape[1] - image_border_width or scale < 1 or scale >= 3:
                    isOutside = True
                    break
                before, curr, after = dogSpace[octave,scale-1:scale+2]
                window = np.stack([before[x-1:x+2,y-1:y+2],
                                   curr[x-1:x+2,y-1:y+2],
                                   after[x-1:x+2,y-1:y+2]]).astype("float32") / 255.
                gradient = Sift.gradient(window)
                hessian = Sift.hessian(window)
                alfa = -np.linalg.lstsq(hessian,gradient,rcond=None)[0]
                if abs(alfa[0]) < 0.5 and abs(alfa[1]) < 0.5 and abs(alfa[2]) < 0.5:
                    break
                y += int(np.round_(alfa[0]))
                x += int(np.round_(alfa[1]))
                scale += int(np.round_(alfa[2]))
                
                
                if x < image_border_width or x >= shape[0] - image_border_width or y < image_border_width or y >= shape[1] - image_border_width or scale < 1 or scale > 3:
                    isOutside = True
                    break
            if isOutside or count >= limit:
                return None
            updated_value = window[1,1,1] + 0.5 * np.dot(gradient,alfa)
            if abs(updated_value) * 3 >= 0.04 :
                xy_hessian = hessian[:2,:2]
                hessian_trace = np.trace(xy_hessian)
                hessian_det = np.linalg.det(xy_hessian)
                if hessian_det > 0 and (((hessian_trace**2)/hessian_det) < (((eigenValue+1)**2)/eigenValue)):
                    keypoint = cv.KeyPoint()
                    keypoint.pt = ((y+alfa[0])*(2 ** octave),(x + alfa[1]) * (2 ** octave))
                    keypoint.octave = octave 
                    keypoint.size = 1 * (2 ** ((scale + alfa[2]) / np.float32(3))) * (2 ** (octave + 1))
                    keypoint.response = abs(updated_value)
                    MyKeyPoint = myKeyPoint(keypoint,2/(2**(octave)),scale)
                    return MyKeyPoint,scale
            return None
                
    def convertSize(keypoints):
        converted = [] 
        for keypoint in keypoints:
            temp = keypoint.keypoint
            temp.pt = tuple(0.5 * np.array(temp.pt))
            temp.size *= 0.5
            keypoint.keypoint = temp
            converted.append(keypoint)
        return converted      
    
            
    def isExtremum(window,threshold):
        center =  window[1,1,1]
        new_ravel = np.zeros((15,))
        new_ravel[:3] = window[:,0,1]
        new_ravel[3:6] = window[:,1,1]
        new_ravel[6:9] = window[:,1,0]
        new_ravel[9:12] = window[:,1,2]
        new_ravel[12:15] = window[:,2,1]
        if abs(center)> threshold:
            #raveled_window = window.ravel()
            return center == max(new_ravel) or center == min(new_ravel)
    
   
    
        
            
    def gradient(window):
        if window.shape[0] < 3 or window.shape[1] < 3 or window.shape[2] < 3: return None
        gs = 0.5 * (window[2,1,1] - window[0,1,1])
        gn = 0.5 * (window[1,2,1] - window[1,0,1])
        gm = 0.5 * (window[1,1,2] - window[1,1,0])
        return np.array([gm,gn,gs]) 
    
    def hessian(window):
        center = window[1,1,1]
        h33 = window[2,1,1] + window[0,1,1] - 2 * center
        h22 = window[1,2,1] + window[1,0,1] - 2 * center
        h11 = window[1,1,2] + window[1,1,0] - 2 * center
        h23 = 0.25 * (window[2,2,1] - window[2,0,1] - window[0,2,1] + window[0,0,1])
        h13 = 0.25 * (window[2,1,2] - window[2,1,0] - window[0,1,2] + window[0,1,0])
        h12 = 0.25 * (window[1,2,2] - window[1,2,0] - window[1,0,2] + window[1,0,0])
         
        return np.array([[h11,h12,h13],
                         [h12,h22,h23],
                         [h13,h23,h33]])
    
    
    def generateDescriptors(keypoints, gaussian_images, window_width=4, num_bins=8, scale_multiplier=3, descriptor_max_value=0.2):
        """Generate descriptors for each keypoint
        """
        descriptors = []

        for keypoint in keypoints:
            temp = keypoint
            keypoint = temp.keypoint
            octave, layer, scale = keypoint.octave, temp.layer, temp.scale
            gaussian_image = gaussian_images[octave, layer]
            num_rows, num_cols = gaussian_image.shape
            point = np.round_(scale * np.array(keypoint.pt)).astype('int')
            bins_per_degree = num_bins / 360.
            angle = 360. - keypoint.angle
            cos_angle = np.cos(np.deg2rad(angle))
            sin_angle = np.sin(np.deg2rad(angle))
            weight_multiplier = -0.5 / ((0.5 * window_width) ** 2)
            row_bin_list = []
            col_bin_list = []
            magnitude_list = []
            orientation_bin_list = []
            tensor = np.zeros((window_width + 2, window_width + 2, num_bins))   # +2 for border
            hist_width = scale_multiplier * 0.5 * scale * keypoint.size
            half_width = int(np.round_(hist_width * np.sqrt(2) * (window_width + 1) * 0.5))   
            half_width = int(min(half_width, np.sqrt(num_rows ** 2 + num_cols ** 2)))    

            for row in range(-half_width, half_width + 1):
                for col in range(-half_width, half_width + 1):
                    row_rot = col * sin_angle + row * cos_angle
                    col_rot = col * cos_angle - row * sin_angle
                    row_bin = (row_rot / hist_width) + 0.5 * window_width - 0.5
                    col_bin = (col_rot / hist_width) + 0.5 * window_width - 0.5
                    if row_bin > -1 and row_bin < window_width and col_bin > -1 and col_bin < window_width:
                        window_row = int(np.round_(point[1] + row))
                        window_col = int(np.round_(point[0] + col))
                        if window_row > 0 and window_row < num_rows - 1 and window_col > 0 and window_col < num_cols - 1:
                            dx = gaussian_image[window_row, window_col + 1] - gaussian_image[window_row, window_col - 1]
                            dy = gaussian_image[window_row - 1, window_col] - gaussian_image[window_row + 1, window_col]
                            gradient_magnitude = np.sqrt(dx * dx + dy * dy)
                            gradient_orientation = np.rad2deg(np.arctan2(dy, dx)) % 360
                            weight = np.exp(weight_multiplier * ((row_rot / hist_width) ** 2 + (col_rot / hist_width) ** 2))
                            row_bin_list.append(row_bin)
                            col_bin_list.append(col_bin)
                            magnitude_list.append(weight * gradient_magnitude)
                            orientation_bin_list.append((gradient_orientation - angle) * bins_per_degree)

            for row_bin, col_bin, magnitude, orientation_bin in zip(row_bin_list, col_bin_list, magnitude_list, orientation_bin_list):
                row_bin_floor, col_bin_floor, orientation_bin_floor = np.floor([row_bin, col_bin, orientation_bin]).astype(int)
                row_fraction, col_fraction, orientation_fraction = row_bin - row_bin_floor, col_bin - col_bin_floor, orientation_bin - orientation_bin_floor
                if orientation_bin_floor < 0:
                    orientation_bin_floor += num_bins
                if orientation_bin_floor >= num_bins:
                    orientation_bin_floor -= num_bins

                c1 = magnitude * row_fraction
                c0 = magnitude * (1 - row_fraction)
                c11 = c1 * col_fraction
                c10 = c1 * (1 - col_fraction)
                c01 = c0 * col_fraction
                c00 = c0 * (1 - col_fraction)
                c111 = c11 * orientation_fraction
                c110 = c11 * (1 - orientation_fraction)
                c101 = c10 * orientation_fraction
                c100 = c10 * (1 - orientation_fraction)
                c011 = c01 * orientation_fraction
                c010 = c01 * (1 - orientation_fraction)
                c001 = c00 * orientation_fraction
                c000 = c00 * (1 - orientation_fraction)

                tensor[row_bin_floor + 1, col_bin_floor + 1, orientation_bin_floor] += c000
                tensor[row_bin_floor + 1, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c001
                tensor[row_bin_floor + 1, col_bin_floor + 2, orientation_bin_floor] += c010
                tensor[row_bin_floor + 1, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c011
                tensor[row_bin_floor + 2, col_bin_floor + 1, orientation_bin_floor] += c100
                tensor[row_bin_floor + 2, col_bin_floor + 1, (orientation_bin_floor + 1) % num_bins] += c101
                tensor[row_bin_floor + 2, col_bin_floor + 2, orientation_bin_floor] += c110
                tensor[row_bin_floor + 2, col_bin_floor + 2, (orientation_bin_floor + 1) % num_bins] += c111

            descriptor_vector = tensor[1:-1, 1:-1, :].flatten()  # Remove histogram borders
            # Threshold and normalize descriptor_vector
            threshold = np.linalg.norm(descriptor_vector) * descriptor_max_value
            descriptor_vector[descriptor_vector > threshold] = threshold
            descriptor_vector /= max(np.linalg.norm(descriptor_vector), float_tolerance)
            # Multiply by 512, round, and saturate between 0 and 255 to convert from float32 to unsigned char (OpenCV convention)
            descriptor_vector = np.round_(512 * descriptor_vector)
            descriptor_vector[descriptor_vector < 0] = 0
            descriptor_vector[descriptor_vector > 255] = 255
            descriptors.append(descriptor_vector)
        return np.array(descriptors, dtype='float32')    




                    

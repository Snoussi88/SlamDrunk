import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform


width = 1920//2
height = 1080//2

class FeatureExtractor(object):
    def __init__(self):
        self.orb = cv2.ORB_create(100)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.last = None


    def extractor(self,img):
        #extraction
        features = cv2.goodFeaturesToTrack(np.mean(img,axis=2).astype(np.uint8), 3000, qualityLevel=0.01, minDistance=3)
        kps = [cv2.KeyPoint(x=f[0][0],y=f[0][1], size=20) for f in features]
        kps, des = self.orb.compute(img,kps)


        #matching similar keypoints (landmarks)
        ret =  []
        if self.last is not None:
            matches = self.bf.knnMatch(des,self.last['des'], k=2)
            for m,n in matches:
                if m.distance < 0.75*n.distance:
                    kp1 = kps[m.queryIdx].pt
                    kp2 = self.last['kps'][m.trainIdx].pt
                    ret.append((kp1,kp2))
            self.last = {'kps': kps, 'des': des}
        else:
            self.last = {'kps': kps, 'des': des}

        #filtering with the fundamental matrix
        if len(ret) > 0:
            ret = np.array(ret)
            print(ret.shape)
            model, inliers = ransac((ret[:,0], ret[:,1]), FundamentalMatrixTransform, min_samples=8, residual_threshold=1, max_trials=100)
            print(sum(inliers))


            ret = ret[inliers]


        return ret


    def process_frame(self,frame):
        img = cv2.resize(frame, (width,height))
        matches  = self.extractor(img)
        if matches is not None:
            for pt1, pt2 in matches:
                u1,v1 = map(lambda x: int(x), pt1)
                u2,v2 = map(lambda x: int(x), pt2)
                cv2.circle(img,(u1,v1), radius=3, color=(0,255,0))
                cv2.line(img,(u1,v1),(u2,v2), color=(255,0,0))
        cv2.imshow("road", img)



cap = cv2.VideoCapture("road.mp4")
if (cap.isOpened() == False):
    print("error reading file, or not found")

fe = FeatureExtractor()


while (cap.isOpened()):
    ret, frame  = cap.read()
    if ret == True:
        fe.process_frame(frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()

    







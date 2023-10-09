import os
import cv2
from tkinter import filedialog
from scipy.spatial import KDTree
import numpy as np
import math

def scalemax(img,maxallow):
    h,w = img.shape[:2]
    maxdim = max([h,w])
    scale = maxallow/maxdim
    resized = cv2.resize(img, (int(round(w*scale)), int(round(h*scale))))
    return resized

def selectcolors(img, maxlen = 1000):
    resized = scalemax(img,maxlen)
    colorpick_points = []
    mask_points = []
    window_name = 'select points'

    def mouse_callback(event, x, y,flags,params):
        nonlocal updated, circolor
        if event == cv2.EVENT_LBUTTONDOWN:
            colorpick_points.append((x, y))
            updated = True
            circolor =[0,255,0]
            return
        if event == cv2.EVENT_RBUTTONDOWN:
            mask_points.append((x, y))
            updated = True
            circolor =[0,0,255]
            return
        
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, mouse_callback)
    updated = False
    circolor = [0,255,0]
    toshow = scalemax(img,maxlen)
    while True:
        cv2.imshow(window_name, toshow)
        key = cv2.waitKey(1)
        if updated:
            if circolor == [0,255,0]:
                center = colorpick_points.copy()[-1]
            else:
                center = mask_points.copy()[-1]
            updated = False
            toshow = cv2.circle(toshow, center, 3, circolor, 2)
        if key in[13,27]:
            cv2.destroyAllWindows()
            break
    colorpick_colors = []
    mask_colors = []
    for i in colorpick_points:
        colorpick_colors.append(list(resized[i[1],i[0]]))
    for i in mask_points:
        mask_colors.append(list(resized[i[1],i[0]]))
    return colorpick_colors, mask_colors

def cartonize(img,landslide_colors,maxallow = 0,export = False):
    cartonized = img.copy()
    if maxallow != 0:
        cartonized = scalemax(img,maxallow)
    tree = KDTree(landslide_colors)
    hc,wc = cartonized.shape[:2]
    for i in range(hc):
        for j in range(wc):
            _, indice = tree.query(cartonized[i,j], k=1)
            # print(cartonized[i,j] , landslide_colors[indice])
            cartonized[i,j] = landslide_colors[indice]
    if export:
        cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)),'prova.png'),cartonized)
    return cartonized

def mergemask(img,mask, export = True, exportname = 'trial.png'):
    output = img.copy()
    for i in range(h):
        for j in range(w):
            if mask[i,j] != 255:
                output[i,j] = np.array([0,255,0])
    if export:
        cv2.imwrite(exportname,output)
    return output

def extract_colors(cartonized):
    monocolors = []
    monocolor_string = []
    for hi in range(len(cartonized)):
        for wi in range(len(cartonized[hi])):
            stringcol = ""
            if len(cartonized.shape)>2:
                for rgb in range(len(cartonized.shape)):
                    stringcol+="_"+str(cartonized[hi,wi][rgb])
            else:
                stringcol="_"+str(cartonized[hi,wi])
            if stringcol[1:] not in monocolor_string:
                monocolor_string.append(stringcol[1:])
                monocolors.append(cartonized[hi,wi])
    return monocolors

def lossless_bgr_to_gray(monocolors, cartonized):
    sorted_colors = sorted([[sum(i),i] for i in monocolors], key=lambda x: x[0])
    gray = np.zeros(cartonized.shape[:2],dtype=np.uint8)
    for hi in range(h):
        for wi in range(w):
            for i in range(len(sorted_colors)):
                if list(sorted_colors[i][1]) == list(cartonized[hi][wi]):
                    gray[hi][wi] = i
    return cv2.equalizeHist(gray)

if __name__ == '__main__':
    newpic = True
    maxlen = 2000
    detail = 100
    pathfile = filedialog.askopenfilename()
    imagename = os.path.basename(pathfile)
    img = scalemax(cv2.imread(pathfile),maxlen)
    h,w = img.shape[:2]
    maxdisplay =[700,1200][[h,w].index(max([h,w]))]
    if newpic:
        monocolors,_ = selectcolors(img,maxlen=maxdisplay)
        cartonized = cartonize(img, monocolors,export=True)
    else:
        cartonized = img.copy()
    monocolors = extract_colors(cartonized)
    print(len(monocolors),'colors found')
    gray = lossless_bgr_to_gray(monocolors,cartonized)
    monograys = extract_colors(gray)
    print(len(monograys),'grays found')
    brush = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(round(min([h,w])/detail),round(min([h,w])/detail)))
    gray =cv2.dilate(gray,brush)
    gray =cv2.erode(gray,brush)
    painted = np.zeros([h,w,3], np.uint8)
    for hi in range(h):
        for wi in range(w):
            painted[hi][wi] = monocolors[monograys.index(gray[hi][wi])]
    cv2.imshow('painted',scalemax(painted,maxdisplay))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)),'painted.png'),painted)

    lines = np.ones([h,w], np.uint8)
    for hi in range(1,h-1):
        for wi in range(1,w-1):
            colorparse=extract_colors(painted[hi-1:hi+1,wi-1:wi+1])
            if len(colorparse)>1:
                cv2.circle(lines,(wi,hi),0,0,1)
    lines =cv2.equalizeHist(lines)
    lines = cv2.flip(lines,1)
    cv2.imshow('only lines',scalemax(lines,maxdisplay))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(os.path.join(os.path.dirname(os.path.abspath(__file__)),'lines.png'),lines.astype(np.uint8))
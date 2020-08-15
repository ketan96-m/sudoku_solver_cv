import cv2
import pytesseract
import matplotlib.pyplot as plt
import numpy as np
import imutils
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model

path = 'data\\'

def draw_grid(img):
    """
    takes a square bgr image 
    converts to grayscale
    evenly divides the image into a grid of 9x9 box
    draw the grid in red colour
    returns the grid points as numpy.array shape (10,10) and
            grayscale image with out grid lines
    """
    copy = img.copy()
    if len(copy.shape)>2:
        copy_gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
    else:
        copy_gray = copy
    side = img.shape[0] #length of the entire square grid
    box_length = side//9 #length of each box
    #get the points along each side to draw a line horizontally and vertically
    arr = list(map(round,np.linspace(0,side,10)))
    pt_horizontal = []
    pt_vertical = []
    #all points as a flat list
    all_pts = []
    for i in arr:
        #top edge
        pt1_top = (int(i),0)
        #bottom edge
        pt2_bottom = (int(i),side)
        #left edge
        pt1_left = (0,int(i))
        #tight edge
        pt2_right = (side,int(i))
        pt_horizontal.append((pt1_top,pt2_bottom))
        pt_vertical.append((pt1_left, pt2_right))
#         cv2.line(copy,pt1_top, pt2_bottom,(0,0,255),5,cv2.LINE_AA)
#         cv2.line(copy,pt1_left, pt2_right,(0,0,255),5,cv2.LINE_AA)
#         cv2.line(copy_gray,pt1_top, pt2_bottom,0,5,cv2.LINE_AA)
#         cv2.line(copy_gray,pt1_left, pt2_right,0,5,cv2.LINE_AA)           
        for j in arr:
            #store the points of each box in the list
            all_pts.append([int(j),int(i)])
            
    cv2.imshow('grid', copy)
    cv2.waitKey()
    cv2.destroyAllWindows()
    if len(copy.shape)>2:
        plt.imshow(copy[:,:,::-1])
    else:
        plt.imshow(copy,cmap='gray')
    all_pts = np.array(all_pts)
    return np.reshape(all_pts,(10,10,-1)), copy_gray

def bounding_box(grid_points):
    """
    takes grid points of all the digit block and returns bounding pts of the box as a list (pt1,pt2)
    returns a list
    """
    bounding_pts = [] #will contain a tuple of pt1 and pt2 of the bounding quadrilateral
    #loop at each row
    for i in range(len(grid_points)-1):
    #loop at each column
        for j in range(len(grid_points)-1):
        #store the point in pt1
            pt1 = grid_points[i,j]
        #store the points next elemnet row-wise and column-wise in pt2
            pt2 = grid_points[i+1,j+1]
        #append both points as tuple in the list bounding_box
            bounding_pts.append((pt1,pt2))
    return bounding_pts

def pre_process(img, skip_dilate = False):
    """
    takes a color image and converts to grayscale, 
    Guassian blur
    adaptive treshold
    returns threshold image
    """
    cop = img.copy()
    if len(cop.shape) > 2:
        cop = cv2.cvtColor(cop, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(cop,(9,9),1,1)
    thres = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,5)
    if skip_dilate == False:
        dilate = cv2.dilate(thres, np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8))
        return dilate
    return thres

def draw_block(pt1,pt2,img,skip_dilate = True):
    """
    returns a digit block from the bounding pts
    """
    img_copy = img.copy()
    draw_img = img_copy[pt1[1]:pt2[1],pt1[0]:pt2[0]]
    if not skip_dilate:
        draw_img = cv2.dilate(draw_img, np.array([[0,1,0],[1,1,1],[0,1,0]],np.uint8))
    return draw_img

def scale_and_center(digit,lf=None, br=None,size=2.5):
    max_area = 0
    seed_point = (None, None)
    h,w = digit.shape[:2]
    if lf == None and br == None:
        margin = int(np.mean((h,w))/size)
        lf = (margin, margin)
        br = (w-margin, h-margin)
    
#     plt.imshow(digit[lf[1]:br[1],lf[0]:br[0]])

    
    for y in range(lf[1],br[1]):
        for x in range(lf[0],br[0]):
            if digit.item(y, x) == 255:
                area = cv2.floodFill(digit,None,(x,y),64)
                if area[0]>max_area:
                    max_area = area[0]
                    seed_point = (x,y)
#     plt.imshow(digit)
#     print(seed_point)
    for x in range(w):
        for y in range(h):
            if digit.item(y, x) == 255 and x < w and y < h:
                cv2.floodFill(digit, None, (x, y), 64)
    mask = np.zeros((h + 2, w + 2), np.uint8)  # Mask that is 2 pixels bigger than the image
    # Highlight the main feature
    if all([p is not None for p in seed_point]):
        cv2.floodFill(digit, mask, seed_point, 255)
    
    top, bottom, left, right = h, 0, w, 0
    
    for x in range(w):
        for y in range(h):
            if digit[y,x] == 64:
                cv2.floodFill(digit,mask, (x,y),0)
                
            if digit.item(y, x) == 255:
                top = y if y < top else top
                bottom = y if y > bottom else bottom
                left = x if x < left else left
                right = x if x > right else right
    
#     print([[left, top], [right, bottom]])
    if top<bottom and left<right:
        bbox = [[left, top], [right, bottom]]   
        digit = digit[top:bottom+1, left:right+1]    
    else:
        digit = np.zeros((h,w),np.uint8)
    return digit


def pad_scale(img,size=28):
    img_copy = img.copy()
    h,w = img.shape[:2]
    digit_max_size = size - 6
    if h>w:
        new_h = digit_max_size
        new_w = int(digit_max_size*w/h)
    else:
        new_w = digit_max_size
        new_h = int(digit_max_size*h/w)
    if new_w > w or new_h > h:
        interpolation = cv2.INTER_CUBIC
    else:
        interpolation = cv2.INTER_AREA
    resize_img = cv2.resize(img_copy,(new_w, new_h), interpolation = interpolation)
    #pad the image at the vertically and sidealong
    #vertical padding
    vert_pad = size - resize_img.shape[0] 
    hor_pad = size - resize_img.shape[1]
    if hor_pad % 2!=0:
        left_pad = hor_pad//2
        right_pad = hor_pad - left_pad
    else:
        left_pad=right_pad = hor_pad//2
    if vert_pad %2!=0:
        top_pad = vert_pad//2
        bottom_pad = vert_pad - top_pad
    else:
        top_pad=bottom_pad = vert_pad//2
#   Ignore any small bounding boxes
    if w > 0 and h > 0 and (w * h) > 100 and len(img_copy) > 0:        
        padded_img = cv2.copyMakeBorder(resize_img,top_pad,bottom_pad,left_pad,right_pad,cv2.BORDER_CONSTANT,0)
#         return scale_and_centre(digit, size, 4)
    else:
        return np.zeros((size, size), np.uint8)
    return padded_img


def digits_padding(img):
    """
    takes a digit block from draw_block() and centers it and pads
    return 28x28 padded image
    """
    if len(img.shape) == 2:
        col = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        col = img.copy()
    digit_cont,_ = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    #find the contour which has the highest aspect ratio
    def aspect_ratio(w,h):
        return max(w,h)/min(w,h)
    
    ap_ratio=[]
    def empty_block():
        return np.zeros((28,28),np.uint8)
    try:
        max_cont_idx1 = np.argmax([cv2.contourArea(i) for i in digit_cont])
        x,y,w,h = cv2.boundingRect(digit_cont[max_cont_idx1])
        max_area = w*h
        if max_area <=100:
            return empty_block()
    except:
        #the block is empty
        return empty_block()
    #crop the image to include only the bounding rectangle
    bound_rect = img[y:y+h,x:x+w]
   
    #final image = 28X28 
    try:
        if max(w,h) >= 21:
            #decrease the image to fit a 21x21 box
            adjust_size = cv2.resize(bound_rect,(int(21/aspect_ratio(w,h)),21),cv2.INTER_AREA)
        elif max(w,h) < 21:
            #increase the image to fit an 21x 21 box
            adjust_size = cv2.resize(bound_rect,(int(21/aspect_ratio(w,h)),21),cv2.INTER_CUBIC)
    except:
        return empty_block()

    vert_pad = 28 - adjust_size.shape[0] 
    hor_pad = 28 - adjust_size.shape[1]
    if hor_pad % 2!=0:
        left_pad = hor_pad//2
        right_pad = left_pad + 1
    else:
        left_pad=right_pad = hor_pad//2
    if vert_pad %2!=0:
        top_pad = vert_pad//2
        bottom_pad = top_pad +1
    else:
        top_pad, bottom_pad = vert_pad//2
        
    padded_img = cv2.copyMakeBorder(adjust_size,top_pad,bottom_pad,left_pad,right_pad,cv2.BORDER_CONSTANT,0)
    #we want an end image of 28x28
    return padded_img

def display_image_grid(bounding_pts,image,skip_dilate=True,size = 2.5):
    """
    construct a grid from the padded images and create the entire grid
    displays the image
    returns an image and list of all the digit images
    """
    if not skip_dilate:
        list_digits = [draw_block(i[0],i[1],image.copy(),skip_dilate = False) for i in bounding_pts]
    else:
        list_digits = [draw_block(i[0],i[1],image.copy()) for i in bounding_pts]
    list_padded_digits = [scale_and_center(i,size=size) for i in list_digits]
    list_padded_digits = [pad_scale(i) for i in list_padded_digits]
    h,w = list_padded_digits[0].shape[:2]
    new_column = 1 
    sudoku_number = []
    for idx, i in enumerate(list_padded_digits):
        #store the digit as a 9 by 9 grid
        if (idx+1)%9 == 1:
            horizontal_stack = i
        else:
            horizontal_stack = np.concatenate((horizontal_stack,i),axis = 1)
        if (idx+1)%9 == 0 and new_column==1:
            vertical_stack = horizontal_stack
            new_column += 1
        elif (idx+1)%9 == 0:
            vertical_stack = np.concatenate((vertical_stack, horizontal_stack),axis = 0)            
#     plt.imshow(vertical_stack, cmap = 'gray')
    return vertical_stack, list_padded_digits

def pred_single(model,img):
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if sum(sum(img))==0:
        return 0
    else:
        #flatten digits
        scale = MinMaxScaler()
        flat = np.reshape(img,(-1))[None,:]
        scale.fit(flat)
        normalize = scale.transform(flat)
        pred = model.predict(normalize)
        return np.argmax(pred)
    
def pred_digits(model,digits):
    #
    scale = MinMaxScaler()
    normalize_list = list(map(scale.fit_transform, digits))
    numbers = []
    for i in normalize_list:
        if sum(sum(i)) == 0:
            numbers.append(0)
        else:
            flat_i = i.reshape((-1))[None,:]
            pred_i = model.predict(flat_i)
            numbers.append(np.argmax(pred_i))
    number_array = np.array(numbers).reshape((9,9))
    return number_array

def get_grid_square(img, skip_dilate = False):
    """
    Take a color image and get the square of the sudoku grid
    """
    zoom_img = img.copy()
    proc_img = pre_process(cv2.cvtColor(zoom_img,cv2.COLOR_GRAY2BGR),skip_dilate = skip_dilate)    
    #find the contours
    contours, heir = cv2.findContours(proc_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key= cv2.contourArea, reverse = True)

    max_area_idx = 0
    color = cv2.cvtColor(zoom_img, cv2.COLOR_GRAY2BGR)
    cv2.imshow('con',cv2.drawContours(color.copy(),contours,max_area_idx,(0,0,255),1,cv2.LINE_AA))
    cv2.waitKey()
    cv2.destroyAllWindows()
    #get four corners of the square
    #
    polygon = contours[max_area_idx]
    
    #
    corner_pts = cv2.approxPolyDP(contours[max_area_idx], 0.009 * cv2.arcLength(contours[max_area_idx], True), True)
    corner_pts = corner_pts.squeeze()
    corner_pts = corner_pts.squeeze()
    bottom_right = np.argmax([i[0]+i[1] for i in corner_pts])
    top_left = np.argmin([i[0]+i[1] for i in corner_pts])
    top_right = np.argmax([i[0]-i[1] for i in corner_pts])
    bottom_left = np.argmin([i[0]-i[1] for i in corner_pts])
    four_corner= [bottom_right,top_left, top_right ,bottom_left]
    pt1 = np.float32([corner_pts[top_left],
                      corner_pts[bottom_left],
                      corner_pts[bottom_right],
                    corner_pts[top_right]])
    #side for the new grid
    side = distance(pt1[0],pt1[1])
    pt2 = np.float32([[0,0],
                    [0,side], 
                    [side,side],
                    [side, 0],      
    ])
    M = cv2.getPerspectiveTransform(pt1, pt2)
    sq_grid = cv2.warpPerspective(proc_img.copy(),M,(side, side))
    return sq_grid
def distance(p1, p2):
    """Returns the scalar distance between two points"""
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return int(np.sqrt((a ** 2) + (b ** 2)))
def pred_conv(img, model):
    #covert the shape of the image to input shape of the DNN
    side = img.shape[0]
    if len(img.shape) != 4:
        if sum(sum(img)) == 0:
            return 0
        else:
            scale = MinMaxScaler()
            scale.fit(img)
            img = scale.transform(img)
            new_shape = img[None,:,:,None]
            pred = model.predict(new_shape)
    return np.argmax(pred)
def pred_conv_com( model,list_digit):
    number = []
    for i in list_digit:
        number.append(pred_conv(i, model))
    number = np.array(number, np.int8)
    return number.reshape((9,9))
def to_string(array):
    return "".join(list(map(str,array.reshape((-1)))))
def show_image(img):
    """Shows an image until any key is pressed"""
    cv2.imshow('image', img)  # Display the image
    cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
    cv2.destroyAllWindows()  # Close all window
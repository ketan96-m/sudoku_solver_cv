import image_processing as im
import sudoku_solver as su
import cv2
from tensorflow.keras.models import load_model
try:
    path = #path of the image
    model = #load_model(<path_of_model>)
    img = cv2.imread(f"{path}4.png",cv2.IMREAD_GRAYSCALE)
    im.show_image(img)
    img = im.get_grid_square(img, skip_dilate=True)
    grid_pts,img = im.draw_grid(img)
    bounding_pts = im.bounding_box(grid_pts)
    img_final, list_digits = im.display_image_grid(bounding_pts,img,skip_dilate=False,size = 3.5)
    array = im.pred_conv_com(model, list_digits)
    print(array)
    array = im.to_string(array)
    su.display(su.solve(array))
except:
    pass
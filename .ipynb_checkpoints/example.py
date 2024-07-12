import cv2
import os
import numpy as np
from skimage import feature
import math
import mysql.connector
# # Đọc tất cả ảnh trong file lưu trữ lại
# Đường dẫn tới thư mục chứa ảnh

def euclidean_distance(vector1, vector2):
    # Chuyển đổi list sang numpy array để sử dụng tính năng của numpy
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    
    # Tính khoảng cách Euclidean
    distance = np.sqrt(np.sum((vector1 - vector2)**2))
    return distance

def read_image(folder_path):
    # Khởi tạo một danh sách để lưu trữ các hình ảnh
    images = []
    
    # Lặp qua tất cả các tệp trong thư mục
    for filename in os.listdir(folder_path):
        # Xác định đường dẫn đầy đủ đến tệp
        file_path = os.path.join(folder_path, filename)
        # Đảm bảo rằng tệp là một tệp ảnh
        if os.path.isfile(file_path) and any(filename.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            # Đọc ảnh và thêm vào danh sách
            image = cv2.imread(file_path)
            if image is not None:
                images.append(image)
    return images



def my_calcHist(image, channels, histSize, ranges):
    # Khởi tạo histogram với tất cả giá trị bằng 0
    hist = np.zeros(histSize, dtype=np.int64)
    # Lặp qua tất cả các pixel trong ảnh
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Lấy giá trị của kênh màu được chỉ định
            bin_vals = [image[i, j, c] for c in channels]
            # Tính chỉ số của bin
            bin_idxs = [(bin_vals[c] - ranges[c][0]) * histSize[c] // (ranges[c][1] - ranges[c][0]) for c in range(len(channels))]
            # Tăng giá trị của bin tương ứng lên 1
            hist[tuple(bin_idxs)] += 1
    return hist



def convert_image_rgb_to_gray(img_rgb):
    # Get the height (h), width (w), and number of channels (_) of the input RGB image
    h, w, _ = img_rgb.shape

    # Create an empty numpy array of zeros with dimensions (h, w) to hold the converted grayscale values
    img_gray = np.zeros((h, w), dtype=np.uint32)

    # Convert each pixel from RGB to grayscale using the formula Y = 0.299R + 0.587G + 0.114B
    for i in range(h):
        for j in range(w):
            r, g, b = img_rgb[i, j]
            gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            img_gray[i, j] = gray_value
            # Return the final grayscale image as a numpy array
    return np.array(img_gray)

def hog_feature(gray_img):  # default gray_image
    # Compute the HOG features and the HOG visualization image using the scikit-image "feature" module's hog() function.
    (hog_feats, hogImage) = feature.hog(
        gray_img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2",
        visualize=True,
    )

    # Return the HOG feature descriptor as a numpy array
    return hog_feats
def feature_extraction(img):  # RGB image
    bins = [16, 16, 16]
    ranges = [[0, 256], [0, 256], [0, 256]]
    hist_my = my_calcHist(img, [0, 1, 2], bins, ranges)
    embedding = np.array(hist_my.flatten()) / (256*256)

    gray_image = convert_image_rgb_to_gray(img)
    embedding_hog = list(hog_feature(gray_image))
    return embedding,embedding_hog

def image_distance(color_hist, hog,color_hist_1 , hog_1):
    color_hist_img1 = color_hist
    hog_img1 = hog
    color_hist_img2 = color_hist_1
    hog_img2 = hog_1
    # khoảng cách sẽ năm trong khoảng từ 0 - căn 2
    color_distance = euclidean_distance(color_hist_img1,color_hist_img2)
    ##chuyển nó về khoảng từ 0-1 bằng cách chia cho căn 2
    color_distance = color_distance/ math.sqrt(2)
    ##Giá trị sẽ nằm trong khoảng từ 0-căn bậc 2 của n
    hog_distance = euclidean_distance(hog_img1,hog_img2)
    ## chuyển về khoảng từ 0-1
    hog_distance =  hog_distance / 186
    #Trả về trung bình khoảng cách
    return (color_distance + hog_distance)/2

# def search_image(image):
#     images = read_image('resize_image')
#     distances = []
#     # Duyệt qua từng ảnh trong mảng images
#     for img in images:     
#         # Tính khoảng cách giữa ảnh hiện tại và ảnh được truyền vào
#         distance = image_distance(img, image )
#         # Thêm tuple chứa khoảng cách và ảnh hiện tại vào danh sách distances
#         print(distance)
#         distances.append((distance, img))
        
#     # Sắp xếp danh sách distances theo khoảng cách tăng dần
#     distances.sort(key=lambda x: x[0])
    
#     # Trả về 5 ảnh có khoảng cách ngắn nhất
#     closest_images = [img for _, img in distances[:5]]
#     return closest_images

def connect_to_database():
    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="tuan462002",
        database="fish"
    )

def search_image(image):
    db = connect_to_database()
    cursor = db.cursor()
    
    # Truy vấn cơ sở dữ liệu để lấy danh sách các ảnh
    cursor.execute("SELECT filename, color_hist, hog FROM image_features")
    images = cursor.fetchall()
    
    distances = []

    color_hist_1 , hog_1 = feature_extraction(image)
    # Duyệt qua từng ảnh trong danh sách images
    for filename, color_hist_str, hog_str in images:
        # Chuyển đổi color_hist và hog từ chuỗi sang numpy array
        color_hist = np.array(list(map(float, color_hist_str.split(','))))
        hog = np.array(list(map(float, hog_str.split(','))))

        # Tính khoảng cách giữa ảnh hiện tại và ảnh được truyền vào
        distance = image_distance(color_hist, hog, color_hist_1 , hog_1 )
        distances.append((distance, filename))
        
    # Sắp xếp danh sách distances theo khoảng cách tăng dần
    distances.sort(key=lambda x: x[0])
    
    # Lấy ra các filename của 5 ảnh có khoảng cách ngắn nhất
    closest_image_filenames = [filename for _, filename in distances[:3]]

    # Trả về danh sách các filename của ảnh
    return closest_image_filenames

def test():
    bgr_image1 = cv2.imread('test.jpg')
    output = search_image(bgr_image1)
    image_folder = "resize_image"
    print(output)
    for filename in output:
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        cv2.imshow(f"{filename}", image)
        cv2.waitKey(0)  # Chờ cho đến khi nhấn một phím bất kỳ
        cv2.destroyAllWindows()

test()



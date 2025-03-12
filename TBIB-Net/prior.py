# # # import os
# # # import cv2

# # # def xiufu(imgpath, maskpath):
# # #     src = cv2.imread(imgpath)  # 读取原始图像
# # #     mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
# # #     # 不进行缩放，直接进行修复操作
# # #     dst = cv2.inpaint(src, mask, 10, cv2.INPAINT_TELEA)
# # #     return dst

# # # # 文件夹路径
# # # image_folder = "/home/jy/zyzcodes1/Polyp-PVT-main/dataset/TrainDataset/images"
# # # mask_folder = "/home/jy/zyzcodes1/Polyp-PVT-main/dataset/TrainDataset/output_folder3"
# # # output_folder = "/home/jy/zyzcodes1/Polyp-PVT-main/dataset/TrainDataset/output_folder4"

# # # # 创建保存修复图像的文件夹
# # # os.makedirs(output_folder, exist_ok=True)

# # # # 遍历图像文件夹中的所有文件
# # # for img_filename in os.listdir(image_folder):
# # #     if img_filename.endswith(".jpg") or img_filename.endswith(".png"):
# # #         img_path = os.path.join(image_folder, img_filename)
# # #         mask_filename = img_filename.replace(".jpg", "_mask.jpg").replace(".png", "_mask.png")
# # #         mask_path = os.path.join(mask_folder, mask_filename)

# # #         if os.path.exists(mask_path):
# # #             repaired_img = xiufu(img_path, mask_path)
# # #             output_path = os.path.join(output_folder, img_filename)
# # #             cv2.imwrite(output_path, repaired_img)
# # #         else:
# # #             print(f"Mask not found for {img_filename}")



# # # import os
# # # import cv2

# # # def xiufu(imgpath,maskpath):
# # #     src_ = cv2.imread(imgpath)
# # #     mask = cv2.imread(maskpath, cv2.IMREAD_GRAYSCALE)
# # #     #缩放因子(fx,fy)
# # #     res_ = cv2.resize(src_,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
# # #     mask = cv2.resize(mask,None,fx=0.6, fy=0.6, interpolation = cv2.INTER_CUBIC)
# # #     dst = cv2.inpaint(res_, mask, 10, cv2.INPAINT_TELEA)
# # #     return dst

# # # # 文件夹路径
# # # image_folder = "/home/jy/zyzcodes1/Polyp-PVT-main/dataset/TrainDataset/images"  # 替换为图像文件夹的实际路径
# # # mask_folder = "/home/jy/zyzcodes1/Polyp-PVT-main/dataset/TrainDataset/output_folder3"    # 替换为掩码文件夹的实际路径
# # # output_folder = "/home/jy/zyzcodes1/Polyp-PVT-main/dataset/TrainDataset/output_folder4"  # 保存修复图像的文件夹路径

# # # # 创建保存修复图像的文件夹
# # # os.makedirs(output_folder, exist_ok=True)

# # # # 遍历图像文件夹中的所有文件
# # # for img_filename in os.listdir(image_folder):
# # #     if img_filename.endswith(".jpg") or img_filename.endswith(".png"):  # 假设您的图像文件是jpg或png格式
# # #         img_path = os.path.join(image_folder, img_filename)
# # #         mask_filename = img_filename.replace(".jpg", "_mask.jpg").replace(".png", "_mask.png")  # 构建对应的掩码文件名
# # #         mask_path = os.path.join(mask_folder, mask_filename)
        
# # #         if os.path.exists(mask_path):
# # #             repaired_img = xiufu(img_path, mask_path)
# # #             output_path = os.path.join(output_folder, img_filename)
# # #             cv2.imwrite(output_path, repaired_img)
# # #         else:
# # #             print(f"Mask not found for {img_filename}")



# #     # default=0.1
# # def __init__(self, top_n=0.10):
# #         self.top_n = top_n

# # def __call__(self, input, x, ratio):
# #         # input: shape [N, 2, H, W]
# #         # x:     shape [N, C, H, W] input image
# #         # ratio: shape[N,]
# #         # f_p: number of foreground pixels (f_p)

# #         # x_u: shape [N, 1, H, W]

# #         # N, C, H, W = x_u.shape
# #         x2 = x.clone()
# #         N, C, H, W = input.shape
# #         for i_c in range(C):

# #             for n in range(N):  # N-dimension index one by one
# #                 f_p = int(ratio[n] * H * W)

# #                 max_list = input[n, i_c, ...]
# #                 max_list = max_list.flatten()
# #                 max_list = max_list.topk(int(f_p * self.top_n))[0]

# #                 if len(max_list)==0:
# #                     max_list_min=1
# #                 else:
# #                     max_list_min = max_list[-1]

# #                 for c in range(x.shape[1]):  # C-dimension index one by one
# #                     x2[n, c, ...][input[n, i_c, ...] > max_list_min] = 0  # input 肯定单channel，因为二分类


# import os
# import cv2
# import findspark




# def Sobel(image):
#     # 1、对X和Y方向求微分
#     x = cv2.Sobel(image, cv2.CV_16S, 1, 0, ksize = 3)
#     y = cv2.Sobel(image, cv2.CV_16S, 0, 1, ksize = 3)
#     #                 深度      x方向阶数 y方向阶数
 
#     # 2、取绝对值
#     absX = cv2.convertScaleAbs(x)  # 转回uint8
#     absY = cv2.convertScaleAbs(y)
 
#     # 3、线性混合
#     dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
#     return dst



# def edges_detection(img):
#     #img must be grayscaled
#     # img_gray = rgb2gray(img) 

#     #partial derivative of x_axis
#     dx = ndimage.sobel(img_gray,1)
#     dy = ndimage.sobel(img_gray,0)
#     #magnitude
#     mag = np.hypot(dx,dy)

#     #shown as image. must be 8 bit integer
#     # mag = 255.0/ np.amax(mag)
#     # mag = mag.astype(np.int8)
#     return mag




# # Path to your dataset folder
# dataset_folder = "/home/jy/datas/ISIC2018/masks/"  # Replace with your dataset folder path

# # Output folder for blended images
# output_folder = "/home/jy/datas/ISIC2018/gtas/"
# os.makedirs(output_folder, exist_ok=True)

# # Get a list of image file names in the dataset folder
# image_file_names = [f for f in os.listdir(dataset_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# for image_file_name in image_file_names:
#     # Load the image from the dataset folder
#     image_path = os.path.join(dataset_folder, image_file_name)
#     image = cv2.imread(image_path, cv2.IMREAD_COLOR)

#     # Convert the image to grayscale
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Apply Canny edge detection
#     edges = cv2.Canny(gray_image, 50, 150)  # You can adjust these threshold values
#     # edges = Sobel(gray_image)

#     # Convert the edge image to a color image

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# # 使用膨胀操作，迭代次数为1
#     edges = cv2.dilate(edges, kernel, iterations=1)

#     edge_mask = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

#     # Blend the edge mask with the original image using addWeighted
#     # alpha = 0.2
#     # blended_image = cv2.addWeighted(image, 1 - alpha, edge_mask, alpha, 0)

#     # Save the blended image to the output folder
#     output_path = os.path.join(output_folder, f'{image_file_name}')
#     cv2.imwrite(output_path, edge_mask)

# print("Batch processing complete.")
# #




import cv2
import numpy as np
import os

# 设置数据集路径
dataset_path = '/home/jy/zyzcodes1/polyp/Polyp-PVT-main/dataset/TrainDataset/masks'
output_path = '/home/jy/zyzcodes1/polyp/Polyp-PVT-main/dataset/TrainDataset/output_folder66'  # 处理后图像的保存路径
os.makedirs(output_path, exist_ok=True)  # 创建输出目录

# 遍历数据集中的每个图像
for filename in os.listdir(dataset_path):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # 检查图像格式
        image_path = os.path.join(dataset_path, filename)
        
        # 读取图像
        image = cv2.imread(image_path)
        
        # 原始代码部分
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold_value = 198
        _, thresholded = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

        # 创建掩码
        mask = thresholded.astype(np.uint8)

        # 找到光亮区域的边界
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建一个新的图像，用于填充光亮区域
        result_image = image.copy()

        # 用周围颜色填充光亮区域
        for contour in contours:
            # 计算光亮区域的外接矩形
            x, y, w, h = cv2.boundingRect(contour)

            # 获取周围像素的平均颜色
            surrounding_region = image[max(0, y - 1):min(y + h + 1, image.shape[0]),
                                 max(0, x - 1):min(x + w + 1, image.shape[1])]

            # 计算周围区域的平均颜色
            surrounding_color = cv2.mean(surrounding_region, mask=None)[:3]

            # 填充光亮区域为周围颜色
            cv2.drawContours(result_image, [contour], -1, surrounding_color, thickness=cv2.FILLED)

        gray = cv2.cvtColor(result_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast_enhanced = clahe.apply(gray)

        # 将对比度增强的灰度图转换为RGB图像
        contrast_enhanced_rgb = cv2.cvtColor(contrast_enhanced, cv2.COLOR_GRAY2BGR)

        # 3. 调整 Canny 算子的阈值进行边缘检测
        edges = cv2.Canny(contrast_enhanced, 50, 150)  # 尝试不同的阈值

        # 多通道融合部分
        # 对每个通道进行边缘检测
        edges_rgb = [cv2.Canny(image[:, :, i], 50, 150) for i in range(3)]  # R, G, B
        edges_hsv = [cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2HSV)[:, :, i], 50, 150) for i in range(3)]  # H, S, V
        edges_lab = [cv2.Canny(cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, i], 50, 150) for i in range(3)]  # L, A, B

        # 最大值融合
        fused_edges = np.maximum.reduce([*edges_rgb, *edges_hsv, *edges_lab])

        # 形态学处理
        kernel = np.ones((3, 3), np.uint8)
        processed_edges = cv2.morphologyEx(fused_edges, cv2.MORPH_CLOSE, kernel)

        # 保存处理后的图像
        output_filename = os.path.join(output_path, filename)
        cv2.imwrite(output_filename, processed_edges)

        print(f'Processed {filename} and saved to {output_filename}')

# 提示处理完成
print("所有图像处理完成！")
import uvicorn
from image_processor import ImageProcessor  # 导入图像处理类
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
import os
from pathlib import Path
import base64
import io
from PIL import Image

app = FastAPI()
processor = ImageProcessor()  # 初始化处理器

# 设备A的结果保存文件夹（固定路径）
DEVICE_A_OUTPUT_FOLDER = "/output/folder"  # 请修改为实际路径


def ensure_folder_exists(folder_path):
    """确保文件夹存在，如果不存在则创建"""
    if not os.path.exists(folder_path):
        try:
            os.makedirs(folder_path)
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"无法创建文件夹：{str(e)}")
    return True


# 确保输出文件夹存在
ensure_folder_exists(DEVICE_A_OUTPUT_FOLDER)


# 定义图片请求模型
class ImageItem(BaseModel):
    filename: str
    image_base64: str  # 图片的Base64编码


# 定义多张图片请求模型
"""
请求时需要写入文件名称和base64码：
 {"images": [
    {
      "images_output":"D:/cc"    ##可选参数，保存路径
      "filename": "0.jpg",
      "image_base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAgAAAQABAAD/..."}]}
"""
class ImagesRequest(BaseModel):
    images: List[ImageItem]
    images_output: Optional[str] = None


def base64_to_image(base64_str):
    """将Base64字符串转换为OpenCV图像"""
    try:
        # 移除可能的前缀
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]

        # 解码Base64字符串
        img_data = base64.b64decode(base64_str)
        # 转换为numpy数组
        nparr = np.frombuffer(img_data, np.uint8)
        # 转换为OpenCV图像
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解码失败：{str(e)}")


@app.post("/process_images/")
async def process_images(request: ImagesRequest):
    """
    接收来自设备B的多张图片
    处理后保存到设备A的固定文件夹中
    """
    if request.images_output:
        output_folder = os.path.join(DEVICE_A_OUTPUT_FOLDER, request.images_output)
        ensure_folder_exists(output_folder)
    else:
        output_folder = DEVICE_A_OUTPUT_FOLDER

    # 处理所有图像
    results = []
    for img_item in request.images:
        try:
            # 将Base64转换为图像
            image = base64_to_image(img_item.image_base64)
            if image is None:
                results.append({
                    "filename": img_item.filename,
                    "status": "失败",
                    "error": "无法解析图像"
                })
                continue

            # 调用图像处理逻辑
            processed_image = processor.process_image(
                image,
                brightness=None,  # 自动亮度调整
                contrast=30,
                shadows=50,
                structure=100
            )

            # 保存处理后的图像到设备A的输出文件夹
            output_file_path = os.path.join(output_folder, img_item.filename)
            cv2.imwrite(output_file_path, processed_image)

            # 转换为Base64返回给设备B
            img_b64 = base64.b64encode(cv2.imencode('.jpg', processed_image)[1]).decode('utf-8')

            results.append({
                "filename": img_item.filename,
                "status": "成功",
                "saved_path": output_file_path,
                "processed_image": img_b64  # 返回处理后的图片Base64
            })
            print(f"\n已处理并保存：{output_file_path}")

        except Exception as e:
            results.append({
                "filename": img_item.filename,
                "status": "失败",
                "error": str(e)
            })

    return {
        "message": f"处理完成，共{len(request.images)}个文件，成功{sum(1 for r in results if r['status'] == '成功')}个",
        "results": results
    }


if __name__ == "__main__":
    uvicorn.run("apt_serve:app", host="0.0.0.0", port=8000, reload=True)

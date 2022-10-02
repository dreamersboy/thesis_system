from django.shortcuts import render
import os
from django.http import HttpResponse
from django.shortcuts import render
import json
from django.http import HttpResponse,JsonResponse
from django.conf import settings
from django.http import HttpResponse
import base64
from . import eval
def index(request):             #添加相关的html页面  templates
    return render(request,"loadpic.html")   #设计一个load的页面
def result(request):
    return render(request,"predict1.html")
def main1(request):
    return render(request,"page1.html")
def files(request):
    # 由前端指定的name获取到图片数据
    img = request.FILES.get('img')
    # 获取图片的全文件名
    img_name = img.name
    # 截取文件后缀和文件名
    #print(img.name)
    mobile = os.path.splitext(img_name)[0]
    ext = os.path.splitext(img_name)[1]
    #print(ext)
    mobile = "test"
    # 重定义文件名
    img_name = f'{mobile}{ext}'
    # 从配置文件中载入图片保存路径
    img_path = os.path.join(settings.IMG_UPLOAD, img_name)
    # 写入文件
    with open(img_path, 'ab') as fp:
        # 如果上传的图片非常大，就通过chunks()方法分割成多个片段来上传
        for chunk in img.chunks():
            fp.write(chunk)

    return render(request,"predict.html")
def search(request):
    if request.is_ajax() and request.method=="POST":
        img_path = os.path.join(settings.IMG_UPLOAD, "test.png")
        print("开始运行eval代码")

        eval.pred()
        print("运行成功")
        return render(request, "predict1.html")
        #return JsonResponse(json.loads(json("ok")))
        # with open(img_path, 'rb') as f:
        #     image_data = f.read()
        #     data=base64.encodebytes(image_data).decode()
        #     src = 'data:image/png;base64,' + str(data)
        #     return JsonResponse(json.loads(src))

def BDNet(request):
    #通过按钮将图片进行预测得到结果

    img_path = os.path.join(settings.IMG_UPLOAD,"test.png")


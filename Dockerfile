# 配置环境
FROM python:3.8
# 工作目录
WORKDIR ./Docker_Python_PCA_KMeans
# 从本来的路径拷贝到容器指定路径，这么写最省事
ADD . .
# 配置python环境库
RUN pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
# 启动后台服务
ENTRYPOINT gunicorn run_predict:app -c gunicorn.conf.py

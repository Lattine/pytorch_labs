
本项目主要用于模型训练过程中Loss日志分析。

------------------
1. 安装visdom模块；
2. 启动。python -m visdom.server；
3. 将格式为"{}\t{}\t{}".format(time, epoch, loss)的数据文件放置losses文件夹下；
4. 运行main.py；
5. 打开浏览器输入http://localhost:8097/
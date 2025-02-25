# 并行工作进程数
workers = 8
# 指定每个工作者的线程数
threads = 1
# 监听内网端口8080
bind = '0.0.0.0:8080'
# 设置守护进程,将进程交给supervisor管理
daemon = 'true'

# 工作模式协程 - 伪线程
# worker_class = 'gevent'
# 设置最大并发量
worker_connections = 1000
# 超时重启
timout = 3600

# 设置进程文件目录
pidfile = './log/gunicorn.pid'
# 设置访问日志和错误信息日志路径
accesslog = './log/gunicorn_access.log'
errorlog = './log/gunicorn_error.log'
# 设置日志记录水平
loglevel = 'warning'

# SSL设置
certfile = './ssl/ssl.pem'
keyfile = './ssl/ssl.key'
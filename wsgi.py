from cas_bird_server import app
from werkzeug.contrib.fixers import ProxyFix

app.wsgi_app = ProxyFix(app.wsgi_app)
app.run(host='0.0.0.0',
        port=8080,
        debug=False,
        ssl_context=('./ssl/birdid.iscas.ac.cn.pem',
                     './ssl/birdid.iscas.ac.cn.key'))

import lutorpy as lua
import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        self.write(returnSay(data["message"]))

    def get(self):
        self.write("Brain!")

if __name__ == "__main__":
    returnSay = require("eval")
    application = tornado.web.Application([
        (r"/", MainHandler),
    ])
    application.listen(80)
    print "Brain started listen on port 80"
    tornado.ioloop.IOLoop.current().start()
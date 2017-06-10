import lutorpy as lua
import json
import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RequestHandler):
    def get(self):

        with open('progress.json') as data_file:
            data = json.load(data_file)
            self.write(data)

if __name__ == "__main__":
    application = tornado.web.Application([
        (r"/", MainHandler),
    ])
    application.listen(80)
    print "Brain started listen on port 80"
    tornado.ioloop.IOLoop.current().start()
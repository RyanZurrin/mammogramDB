import time
import tornado
import tornado.ioloop
import tornado.web
import tornado.gen
import tornado.concurrent
import tornado.process
import json
from concurrent.futures import ThreadPoolExecutor


class Core:
    """
    The core can be anything outside of tornado!
    """

    def get(self, remote_ip):
        # print('going to sleep', remote_ip)
        # time.sleep(10)
        # print('done sleeping', remote_ip)

        with open("roi.html", "r") as f:
            html = f.read()

        return html


class MainHandler(tornado.web.RequestHandler):
    """
    Our custom request handler who implements a co-routine
    and forwards the calls to an external class instance (core).
    """

    def initialize(self, executor, core, webserver):
        self._executor = executor
        self._core = core
        self._webserver = webserver

    @tornado.gen.coroutine
    def get(self, uri, a):
        """
        This method has to be decorated as a coroutine!
        """
        ip = self.request.remote_ip

        print(self.request.uri)
        print(uri)

        if uri == "/ryan":
            # any connection to O.DeepSight..
            print("sth")

            a = json.loads(self.request.query)

            print(a)
            self.write("dfsdfsdf")

        #
        # yield is important here
        # and obviously, the executor!
        #
        # we connect the get handler now to the core
        #
        res = yield self._executor.submit(self._core.get, remote_ip=ip)

        self.write(res)


class WebServer:
    def __init__(self, port=8888):
        """ """
        self._port = port

    def start(self, core):
        """ """
        # the important part here is the ThreadPoolExecutor being
        # passed to the main handler, as well as an instance of core
        webapp = tornado.web.Application(
            [
                (
                    r"(/(.*))",
                    MainHandler,
                    {
                        "executor": ThreadPoolExecutor(max_workers=10),
                        "core": core,
                        "webserver": self,
                    },
                )
            ]
        )
        webapp.listen(self._port)
        tornado.ioloop.IOLoop.current().start()


#
# THE ENTRY POINT
#

# ```bash
# for V in 1 2 3 4 5 6 7 8 9 10 11 12 13
# do
#   curl -i http://localhost:8888/ &
# done
# ````

# run the webserver
webserver = WebServer()
webserver.start(Core())

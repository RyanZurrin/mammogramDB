import sys

sys.path.append("..")
import omama as O
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
import os
import tornado
import tornado.concurrent
import tornado.gen
import tornado.ioloop
import tornado.process
import tornado.web


class Core:
    """
    The core can be anything outside of tornado!
    """

    def get(self, remote_ip, uri):
        """
        Get the data from the server.
        Parameters
        ----------
        remote_ip : str
            The IP address of the server.
        uri : str
            The URI of the data.

        Returns
        -------
        data : str
            The data from the server.

        """
        if uri == "/roi":
            with open("roi.html", "r") as f:
                html = f.read()
            return html
        elif uri == "/get_2d":
            t0 = time.time()
            # call the DataHelper API to get the 2d data
            print("getting a 2d dicom from datahelper")
            img = O.DataHelper.get2D(config_num=2, randomize=True)
            print("running dicom through deepsight")
            pred = O.DeepSight.run(img)
            print("prediction: ", pred)
            # dh.view(img[0], pred)
            # send image to the roi frontend
            print("saving image for loading to roi frontend")
            # filename = img[0].SOPInstanceUID + '.dcm'
            # O.DataHelper.store(img[0], filename)
            print("time:", time.time() - t0)
            return img[0]
        elif uri == "/get_3d":
            t0 = time.time()
            # call the DataHelper API to get the 3d data
            print("getting a 3d dicom from datahelper")
            img = O.DataHelper.get3D(config_num=1, randomize=True)
            print("running dicom through deepsight")
            pred = O.DeepSight.run(img)
            print("prediction: ", pred)
            # dh.view(img[0], pred)
            # send image to the roi frontend
            print("time:", time.time() - t0)
            return img[0]
        elif uri == "/ldfs":
            with open("ldfs.html", "r") as f:
                html = f.read()
            return html
        else:
            if uri.endswith(".html"):
                with open(uri, "r") as f:
                    data = f.read()
                return data


class MainHandler(tornado.web.RequestHandler):
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
        #
        # yield is important here
        # and obviously, the executor!
        #
        # we connect the get handler now to the core
        #
        res = yield self._executor.submit(self._core.get, remote_ip=ip, uri=uri)
        # if uri ends in .html
        if uri.endswith(".html"):
            self.write(res)
        else:
            if uri == "/roi" or uri == "/ldfs":
                self.write(res)
            else:
                print(res)


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
            ],
            debug=True,
            autoreload=True,
        )
        webapp.listen(self._port)
        tornado.ioloop.IOLoop.current().start()

    @property
    def port(self):
        """returns the port"""
        return self._port


# run the webserver
webserver = WebServer()
webserver.start(Core())

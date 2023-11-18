import sys

sys.path.append("..")
import omama as O
import getpass
import datetime
import tornado
import tornado.ioloop
import tornado.web
import tornado.gen
import tornado.concurrent
import tornado.process
import json
import random
from concurrent.futures import ThreadPoolExecutor
import os
from io import BytesIO

debug = False  # set to True to see debug messages

# --------------------- Global Variables Need to be Set ------------------------
PORT = 8888
CACHE_DIR = "/raid/mpsych/OMAMA/DATA/cache_files/roi_user_cachefiles/"
MAIN_CACHE_PATH = CACHE_DIR + "roi_main_cache.json"
USER = getpass.getuser()
USER_CACHE_PATH = CACHE_DIR + USER + ".json"
LOCAL_SAVE_PATH = "/tmp/coords.json"
LOCAL_CACHE_PATH = "test_dicoms/"


# --------------------------- End of Global Variables --------------------------


class Core:
    """The Core can be anything outside the Tornado framework."""

    def get(self, request):
        """This is the main function of the Tornado framework.

        Parameters
        ----------
        request : tornado.httputil.HTTPServerRequest
            The request object.

        Returns
        -------
        str
            The response string.
        """
        command_line_args = sys.argv
        local_flag = False
        config_num = 2
        if "--local" in command_line_args:
            local_flag = True
        if "--config" in command_line_args:
            config_num = int(command_line_args[command_line_args.index("--config") + 1])
        if debug:
            print("request:", request, "type:", type(request), "local:", local_flag)
        return self._load(request, config_num, local_flag)

    @staticmethod
    def _write_predictions_to_cache(path, sop_uid, predictions):
        """Write the predictions to the cache file.

        Parameters
        ----------
        path : str
            The path to the cache file.
        sop_uid : str
            The SOP UID of the image.
        predictions : dict
            The predictions to write to the cache file.
        """

        if debug:
            print("writing predictions to cache")
        if not os.path.isfile(path):
            with open(path, "w") as f:
                pred = {sop_uid: [predictions]}
                json.dump(pred, f)
        else:
            with open(path, "r") as f:
                roi_json = json.load(f)
            if sop_uid in roi_json:
                prev = roi_json[sop_uid]
                prev.append(predictions)
                roi_json[sop_uid] = prev
            else:
                roi_json[sop_uid] = [predictions]

            with open(path, "w") as f:
                json.dump(roi_json, f)
        if debug:
            print("done writing predictions to cache")

    @staticmethod
    def _load(request, config_num, local_flag):
        """Load the data from the source.

        Parameters
        ----------
        request : tornado.httputil.HTTPServerRequest
            The request object.
        local_flag : bool, optional
            Whether to load the data from the local cache or the remote server.
            The default is False.

        Returns
        -------
        str
            The response string.
        """
        if debug:
            print("in load")
        args = request.arguments
        if request.uri.startswith("/get_next_dicom"):
            if local_flag:
                all_dicoms = os.listdir(LOCAL_CACHE_PATH)
                which = int(len(all_dicoms) * random.random())
                current = all_dicoms[which].replace(".dcm", "")
                return '{"msg":"' + current + '"}'
            else:
                img = O.DataHelper.get2D(config_num=config_num, randomize=True)
                sop_uid = img[0].SOPInstanceUID
                if debug:
                    print("current:", sop_uid)
                return '{"msg":"' + sop_uid + '"}'

        elif request.uri.startswith("/get_dicom"):
            if debug:
                print("getting a dicom")
            if "id" in args:
                sop_id = args["id"][0].decode("utf-8")
            else:
                return '{"msg":"no id"}'
            if local_flag:
                if debug:
                    print("getting local dicom")
                with open("test_dicoms/" + sop_id + ".dcm", "rb") as f:
                    output = BytesIO()
                    output.write(f.read())
                    content = output.getvalue()
                    return content  # is type bytes
            else:
                if debug:
                    print("getting remote dicom")
                img = O.DataHelper.get(sop_id)
                path = img.filePath
                if debug:
                    print("path:", path)
                    print("sop_id:", sop_id)
                with open(path, "rb") as f:
                    output = BytesIO()
                    output.write(f.read())
                    content = output.getvalue()
                    return content  # is type bytes

        elif request.uri.startswith("/roi"):
            # get the date time stamp to the millisecond
            date_time = datetime.datetime.now()
            date_time_stamp = date_time.strftime("%Y-%m-%d_%H-%M-%S-%f")
            x1 = args["x1"][0].decode("utf-8")
            y1 = args["y1"][0].decode("utf-8")
            x2 = args["x2"][0].decode("utf-8")
            y2 = args["y2"][0].decode("utf-8")
            sop_uid = args["id"][0].decode("utf-8")
            score = args["score"][0].decode("utf-8")
            quality = args["quality"][0].decode("utf-8")
            if debug:
                print(
                    f" sop_uid:{sop_uid}, x1:{x1}, y1:{y1}, x2:{x2}, "
                    f"y2:{y2}, score:{score}, quality:{quality}, "
                    f"user:{USER}, date_time:{date_time_stamp}"
                )

            pred = {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "score": score,
                "quality": quality,
                "user": USER,
                "date_time": date_time_stamp,
            }

            if local_flag:
                Core._write_predictions_to_cache(LOCAL_SAVE_PATH, sop_uid, pred)
                return '{"msg":"ok"}'
            else:
                Core._write_predictions_to_cache(USER_CACHE_PATH, sop_uid, pred)
                Core._write_predictions_to_cache(MAIN_CACHE_PATH, sop_uid, pred)
                return '{"msg":"ok"}'

        elif request.uri.endswith("favicon.ico"):
            # display the favicon
            with open("images/favicon.ico", "rb") as f:
                return f.read()
        elif request.uri.endswith("script.js"):
            with open("js/script.js", "r") as f:
                js = f.read()
            if debug:
                print("returning js")
            return js
        elif request.uri.endswith("style.css"):
            with open("css/style.css", "r") as f:
                style = f.read()
            if debug:
                print("returning css")
            return style
        else:
            with open("index.html", "r") as f:
                html = f.read()
            if debug:
                print("returning html")
            return html


class MainHandler(tornado.web.RequestHandler):
    """Our custom request handler who implements a co-routine and forwards the
    calls to an external class instance (core).

    Parameters
    ----------
    tornado.web.RequestHandler
        The base class.
    """

    def initialize(self, executor, core, web_server):
        self._executor = executor
        self._core = core
        self._webserver = web_server

    @tornado.gen.coroutine
    def get(self, uri, a):
        """
        This method has to be decorated as a coroutine!

        Parameters
        ----------
        uri : str
            The URI of the request.
        a : str
            The argument of the request.

        Returns
        -------
        self.write(response)
            The response to the request.
        """
        #
        # yield is important here
        # and obviously, the executor!
        #
        # we connect the get handler now to the core
        #
        res = yield self._executor.submit(self._core.get, request=self.request)
        self.write(res)
        self.finish()


class WebServer:
    def __init__(self, port=PORT):
        """Initialize the web server.

        Parameters
        ----------
        port : int
            The port to use. The default is PORT.
        """
        self._port = port

    def start(self, core):
        """Start the web server.

        Parameters
        ----------
        core : Core
            The core instance.

        Returns
        -------
        tornado.web.Application
            The web server instance.
        """
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
                        "web_server": self,
                    },
                )
            ],
            debug=True,
            autoreload=True,
        )

        print("Running on http://localhost:" + str(self._port))
        webapp.listen(self._port)
        tornado.ioloop.IOLoop.current().start()

    @property
    def port(self):
        """returns the port"""
        return self._port


# run the webserver on the local machine
webserver = WebServer()
webserver.start(Core())

"""Log

Log functionalities for Python projects or scripts.
File source used only as module. You can run this file source as script only to test the functionality implemented here.

Attributes
----------
LOG_FMT : str
    Default string format for logs
LOG_DATE_FMT : str
    Default string date format for logs

Classes
-------
Log()
    class for log on console and file, using "logging" package

Functions
---------
test()
    a brief example of the usage of the functionalities implemented in this py file
"""

import logging
import sys

# Constants for log formatting
LOG_FMT = "[%(asctime)s] %(name)s - %(levelname)s: %(message)s"
LOG_DATE_FMT = "%d/%m/%Y %H:%M:%S"


class Log:
    """
    Class for log on console and file.

    Attributes
    ----------
    logger : Logger
        logger library instance
    name : str
        module name
    lvl : logging constant
        Threshold of debug filtered
    enable_console : bool
        True if log print on console, False if not
    enable_file : bool
        True if log written in file, False if not
    filename : str
        Name of file to write logs

    Methods
    -------
    f(fmt, ...)
        fatal log message formatted as printf style
    e(fmt, ...)
        error log message formatted as printf style
    w(fmt, ...)
        warning log message formatted as printf style
    i(fmt, ...)
        info log message formatted as printf style
    d(fmt, ...)
        debug log message formatted as printf style
    """

    def __init__(self, name, lvl=logging.DEBUG, enable_console=True, enable_file=False, filename=None):
        """Log class initializer

        Parameters
        ----------
        name : str
            module name
        lvl : logging constant
            Threshold of debug filtered
        enable_console : bool
            True if log print on console, False if not
        enable_file : bool
            True if log written in file, False if not
        filename : str
            Name of file to write logs
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(lvl)
        self.logger.addFilter(logging.Filter(""))
        self.name = name
        self.lvl = lvl
        self.enable_console = enable_console
        self.enable_file = enable_file

        if not filename and enable_file:
            self.filename = name
        else:
            self.filename = filename

        # Manage console log.
        if enable_console:
            sh = logging.StreamHandler(stream=sys.stdout)
            sh_format = logging.Formatter(fmt=LOG_FMT, datefmt=LOG_DATE_FMT)
            sh_filter = logging.Filter("")
            sh.setFormatter(sh_format)
            sh.addFilter(sh_filter)
            self.logger.addHandler(sh)

        # Manage console log.
        if enable_file:
            fh = logging.FileHandler(self.filename + ".log")
            fh_format = logging.Formatter(fmt=LOG_FMT, datefmt=LOG_DATE_FMT)
            fh_filter = logging.Filter("")
            fh.setFormatter(fh_format)
            fh.addFilter(fh_filter)
            self.logger.addHandler(fh)

    def __log(self, t, msg, *args, **kwargs):
        """Log message print

        Parameters
        ----------
        t : logging type
            one of the following: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL.
        msg : str
            message formatted in printf style
        args : list
            arguments to inject in format message
        kwargs : list
            other parameter for logging library
        """
        if t == logging.CRITICAL:
            self.logger.fatal(msg, *args, **kwargs)
        elif t == logging.ERROR:
            self.logger.error(msg, *args, **kwargs)
        elif t == logging.WARNING:
            self.logger.warning(msg, *args, **kwargs)
        elif t == logging.INFO:
            self.logger.info(msg, *args, **kwargs)
        elif t == logging.DEBUG:
            self.logger.debug(msg, *args, **kwargs)

    def f(self, msg, *args, **kwargs):
        """Fatal log message"""
        self.__log(logging.CRITICAL, msg, *args, **kwargs)

    def e(self, msg, *args, **kwargs):
        """Error log message"""
        self.__log(logging.ERROR, msg, *args, **kwargs)

    def w(self, msg, *args, **kwargs):
        """Warning log message"""
        self.__log(logging.WARNING, msg, *args, **kwargs)

    def i(self, msg, *args, **kwargs):
        """Info log message"""
        self.__log(logging.INFO, msg, *args, **kwargs)

    def d(self, msg, *args, **kwargs):
        """Debug log message"""
        self.__log(logging.DEBUG, msg, *args, **kwargs)

    def set_level(self, lvl):
        """Update the logger log level

        Parameters
        ----------
        lvl : logging level
            one of the following: logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL.
        """
        if lvl in [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]:
            self.logger.setLevel(lvl)


def test():
    """Test function call if this py file is run as script"""
    log = Log()
    log.f("test log fatal %i", 1)
    log.e("test log error %i", 2)
    log.w("test log warning %i", 3)
    log.i("test log info %i", 4)
    log.d("test log debug %i", 5)


if __name__ == '__main__':
    test()




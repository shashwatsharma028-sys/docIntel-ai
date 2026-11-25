"""All PyMuPDF work runs on ONE dedicated thread.

PyMuPDF does not support multithreaded use (pymupdf docs, "Multiprocessing" recipe). The API
uses a thread pool for request handling, so PDF parsing/validation is funnelled through a
single-worker executor: PyMuPDF is only ever called from that thread, while chat requests
(which never touch PyMuPDF) stay concurrent.
"""
import threading
from concurrent.futures import ThreadPoolExecutor

_PREFIX = "pymupdf-worker"
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=_PREFIX)


def run_pdf(fn, *args, **kwargs):
    if threading.current_thread().name.startswith(_PREFIX):
        return fn(*args, **kwargs)
    return _executor.submit(fn, *args, **kwargs).result()

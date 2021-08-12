import queue
from multiprocessing.managers import BaseManager


class ReceiverMessager(object):
    """
    ReceiverMessager for worker to receive additional message from master
    message queue is on receiver
    """
    def __init__(self, ip="", port=13579, authkey=b'abc', max_len=100):
        self.ip = ip
        self.port = port
        self.authkey = authkey
        self.max_queue_length = max_len
        self.queue = None
        self.manager = None
        self._init_receiver()

    def _init_receiver(self):
        _queue = queue.Queue(maxsize=self.max_queue_length)
        QueueManager.register('get_queue', callable=lambda: _queue)
        self.manager = QueueManager(address=(self.ip, self.port), authkey=self.authkey)
        self.manager.start()
        self.queue = self.manager.get_queue()

    def send_message(self, message):
        raise NotImplementedError

    def receive_message(self):
        if self.queue.empty() is True:
            return None
        message = self.queue.get()
        return message

    # def shutdown(self):
    #     self.manager.shutdown()


class QueueManager(BaseManager):
    pass

from multiprocessing.managers import BaseManager


class SenderMessager(object):
    """
    SenderMessager for master to send additional message to worker
    message queue is on receiver
    """
    def __init__(self, ip="127.0.0.1", port=13579, authkey=b'abc'):
        self.ip = ip
        self.port = port
        self.authkey = authkey
        self.queue = None
        self.manager = None
        self._init_sender()

    def _init_sender(self):
        QueueManager.register('get_queue')
        self.manager = QueueManager(address=(self.ip, self.port), authkey=self.authkey)
        self.manager.connect()
        self.queue = self.manager.get_queue()

    def send_message(self, message):
        self.queue.put(message)

    def receive_message(self):
        raise NotImplementedError

    # def shutdown(self):
    #     raise NotImplementedError


class QueueManager(BaseManager):
    pass

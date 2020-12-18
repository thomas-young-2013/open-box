# This file is adapted from github repo: https://github.com/automl/HpBandSter.
import os
import threading
import configparser
import Pyro4.naming


class NameServer(object):
	"""
	The nameserver serves as a phonebook-like lookup table for your workers. Unique names are created so the workers
	can work in parallel and register their results without creating racing conditions. The implementation uses
	`PYRO4 <https://pythonhosted.org/Pyro4/nameserver.html>`_ as a backend and this class is basically a wrapper.
	"""
	def __init__(self, run_id, host=None, port=0, config_directory=None):
		"""
		Parameters
		----------
			run_id: str
				unique run_id associated with the HPB run
			working_directory: str
				path to the working directory of the HPB run to store the nameservers credentials.
				If None, no config file will be written.
			host: str
				the hostname to use for the nameserver
			port: int
				the port to be used. Default (=0) means a random port
		"""
		self.run_id = run_id
		self.host = host
		self.port = port
		self.dir = config_directory
		self.conf_fn = None
		self.pyro_ns = None
		self.init_config()

	def init_config(self):
		if self.dir is None:
			self.dir = 'conf'
		config_path = os.path.join(self.dir, 'distrib.config')
		config = configparser.ConfigParser()
		config.read(config_path)
		name_server = dict(config.items('name_server'))
		self.host = name_server['nameserver']
		if self.host == '127.0.0.1':
			self.host = 'localhost'
		self.port = int(name_server['nameserver_port'])

	def start(self):
		"""	
		starts a Pyro4 nameserver in a separate thread
		
		Returns
		-------
			tuple (str, int):
				the host name and the used port
		"""

		uri, self.pyro_ns, _ = Pyro4.naming.startNS(host=self.host, port=self.port)

		self.host, self.port = self.pyro_ns.locationStr.split(':')
		self.port = int(self.port)
		
		thread = threading.Thread(target=self.pyro_ns.requestLoop, name='Pyro4 nameserver started by Lite-BO')

		thread.start()

		return self.host, self.port

	def shutdown(self):
		"""
			clean shutdown of the nameserver and the config file (if written)
		"""
		if self.pyro_ns is not None:
			self.pyro_ns.shutdown()
			self.pyro_ns = None
		
		if self.conf_fn is not None:
			os.remove(self.conf_fn)
			self.conf_fn = None

	def __del__(self):
		self.shutdown()




import zmq


class Forwarder:
    def __init__(self, frontendPort: str, backendPort: str):
        self.frontendPort = frontendPort
        self.backendPort = backendPort
        self.frontend = None
        self.backend = None

    def connect(self):
        context = zmq.Context(1)
        # Socket facing clients (subscribers)
        self.frontend = context.socket(zmq.SUB)
        self.frontend.bind("tcp://192.168.100.143:" + self.frontendPort)
        self.frontend.setsockopt_string(zmq.SUBSCRIBE, str(''))
        # Socket facing services (Publishers)
        self.backend = context.socket(zmq.PUB)
        self.backend.bind("tcp://192.168.100.143:" + self.backendPort)
        print("connected...")
        zmq.device(zmq.FORWARDER, self.frontend, self.backend)
        print("connected...")

    def close(self):
        self.frontend.close()
        self.backend.close()
        self.context.term()

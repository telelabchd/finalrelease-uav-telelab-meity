from Forwarder import Forwarder


def start():
    print("hello 2")
    # Ml feed
    forwarder2 = Forwarder("5567", "5568")
    forwarder2.connect()
    print("started ml feed")

start()
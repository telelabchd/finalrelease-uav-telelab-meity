from Forwarder import Forwarder


def start():
    print("hello 1")
    forwarder1 = Forwarder("5565", "5566")
    forwarder1.connect()
    print("started basic feed")
    # Ml feed


start()
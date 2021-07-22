import sys
import Pyro4

#sys.excepthook = Pyro4.util.excepthook

class SubscriberPyro(object):
    def __init__(self, name):
        self.name = name

    def visit(self, warehouse):
        print("This is {0}.".format(self.name))
        self.callROS(warehouse)
        print("Thank you, come again!")

    def callROS(self, warehouse):
        warehouse.connection_point()

if __name__ == '__main__':
    ROSside = Pyro4.Proxy("PYRONAME:ros_side.publisherpyro")
    connect_ros = SubscriberPyro("nehildanis") 
    connect_ros.visit(ROSside)

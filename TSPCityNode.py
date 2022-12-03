import time
import numpy as np
from TSPClasses import *
import heapq
import itertools

class TSPCityNode:
    def __init__(self, city):
        self.city: City = city
        self.outGoingNodes = []
        self.forwardConnection = None
        self.backwardConnection = None
        #TODO: find outgoing and incoming with matrix

    def getIndex(self):
        return self.city.getIndex()

    def getCity(self):
        return self.city

    def getForwardConnection(self):
        return self.forwardConnection

    def getBackwardConnection(self):
        return self.backwardConnection

    def setForwardConnection(self, connectedNode):
        self.forwardConnection = connectedNode

    def setBackwardConnection(self, connectedNode):
        self.backwardConnection = connectedNode

    def addOutGoingNode(self, node):
        self.outGoingNodes.append(node)

    def getOutGoing(self):
        return self.outGoingNodes


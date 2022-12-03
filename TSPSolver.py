#!/usr/bin/python3
from TSPCityNode import TSPCityNode
from which_pyqt import PYQT_VER

if PYQT_VER == 'PYQT5':
    from PyQt5.QtCore import QLineF, QPointF
else:
    raise Exception('Unsupported Version of PyQt: {}'.format(PYQT_VER))

import time
import numpy as np
from TSPClasses import *
import heapq
import itertools


class TSPSolver:
    def __init__(self, gui_view):
        self._scenario = None

    def setupWithScenario(self, scenario):
        self._scenario = scenario

    ''' <summary>
        This is the entry point for the default solver
        which just finds a valid random tour.  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of solution, 
        time spent to find solution, number of permutations tried during search, the 
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def defaultRandomTour(self, time_allowance=60.0):
        results = {}
        cities = self._scenario.getCities()
        ncities = len(cities)
        foundTour = False
        count = 0
        bssf = None
        start_time = time.time()
        while not foundTour and time.time() - start_time < time_allowance:
            # create a random permutation
            perm = np.random.permutation(ncities)
            route = []
            # Now build the route using the random permutation
            for i in range(ncities):
                route.append(cities[perm[i]])
            bssf = TSPSolution(route)
            count += 1
            if bssf.cost < np.inf:
                # Found a valid route
                foundTour = True
        end_time = time.time()
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - start_time
        results['count'] = count
        results['soln'] = bssf
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the greedy solver, which you must implement for 
        the group project (but it is probably a good idea to just do it for the branch-and
        bound project as a way to get your feet wet).  Note this could be used to find your
        initial BSSF.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found, the best
        solution found, and three null values for fields not used for this 
        algorithm</returns> 
    '''

    def greedy(self, time_allowance=60.0):
        # instantiate matrix - O(n^2)
        cities = self._scenario.getCities()
        self.makeMatrix(cities)
        # make cities into dictionary - O(n)
        citiesDict = {index: city for index, city in enumerate(cities)}
        nCities = len(cities)
        iterator = 0
        foundTour = False
        bssf = None
        startTime = time.time()
        #
        while not foundTour and time.time() - startTime < time_allowance and iterator < nCities:
            # copy the list O(n)
            startCity = cities[iterator]
            citiesToVisit = citiesDict.copy()
            currentCity = startCity
            potentialRoute = [citiesToVisit.pop(currentCity._index)]
            # for each city check for next path to take - O(n^2)
            while citiesToVisit:
                closestCity = (None, math.inf)
                # iterate over each city to visit and check if it's the closest - O(n)
                for index in citiesToVisit:
                    city = citiesToVisit.get(index)
                    if currentCity.costTo(city) < closestCity[1]:
                        closestCity = (index, currentCity.costTo(city))
                currentCity = citiesToVisit.get(closestCity[0])
                if currentCity is None: break
                potentialRoute.append(citiesToVisit.pop(currentCity._index))

            if not citiesToVisit:
                foundTour = True
                bssf = TSPSolution(potentialRoute)
            iterator += 1
        end_time = time.time()

        results = {}
        results['cost'] = bssf.cost if foundTour else math.inf
        results['time'] = end_time - startTime
        results['count'] = iterator
        results['soln'] = bssf if bssf.cost < math.inf else None
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    ''' <summary>
        This is the entry point for the branch-and-bound algorithm that you will implement
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number solutions found during search (does
        not include the initial BSSF), the best solution found, and three more ints: 
        max queue size, total number of states created, and number of pruned states.</returns> 
    '''

    def branchAndBound(self, time_allowance=60.0):
        pass

    ''' <summary>
        This is the entry point for the algorithm you'll write for your group project.
        </summary>
        <returns>results dictionary for GUI that contains three ints: cost of best solution, 
        time spent to find best solution, total number of solutions found during search, the 
        best solution found.  You may use the other three field however you like.
        algorithm</returns> 
    '''
    def convertCitiesToCityNodes(self, cities):
        cityNodes = []
        for city in cities:
            cityNodes.append(TSPCityNode(city))
        return cityNodes

    # make a new matrix with the custom cityNodes - O(n^2)
    def setOutGoing(self, cities, matrix):
        # add each node to city - O(n)
        nodeDict = {}
        for cityNode in cities:
            nodeDict.update({cityNode.getIndex(): cityNode})
        # for each node, add connections to new list - O(n^2)
        for cityNode in cities:
            outGoing = matrix[cityNode.getIndex(), :] #takes whole row
            # add each outgoing to the new dictionary - O(n)
            for i in range(len(outGoing)):
                if outGoing[i] < math.inf:
                    cityNode.addOutGoingNode(nodeDict.get(i))

    def getStarterCities(self, cities, failedPairs, tour):
        # make the base tour with city nodes - O(1)
        node1 = None
        node2 = None
        cost = math.inf
        for city in cities:
            for outGoing in city.getOutGoing():
                if (city, outGoing) in failedPairs:
                    continue
                if city.getCity().costTo(outGoing.getCity()) < cost:
                    node1 = city
                    node2 = outGoing
                    cost = city.getCity().costTo(outGoing.getCity())

        node1.setForwardConnection(node2)
        node1.setBackwardConnection(node2)
        node2.setForwardConnection(node1)
        node2.setBackwardConnection(node1)
        tour.append(node1)
        tour.append(node2)
        return node1, node2


    # iteratively
    def fancy(self, time_allowance=60.0):
        # instantiate the distance matrix - O(n^2)
        nonNodeCities = self._scenario.getCities()
        ncities = len(nonNodeCities)
        matrix = self.makeMatrix(nonNodeCities)

        # convert scenario cities into city nodes - O(n^2)
        cities = self.convertCitiesToCityNodes(self._scenario.getCities())
        self.setOutGoing(cities, matrix)

        startTime = time.time()

        # until we have a complete tour, find the cheapest city to add to the tour and add it.
        # Complexity is O(n) * (O(n^4 + n)) = O(n^5)
        failedPairs = []
        solutionFound = False
        while not solutionFound:
            tour = []
            failedTour = False
            startCity1, startCity2 = self.getStarterCities(cities, failedPairs, tour)
            while not failedTour:
                cheapestCity, cheapestOrigin, cheapestInsertion = self.findClosestNeighborNotInTour(tour, matrix)
                if cheapestInsertion == math.inf:
                    failedPairs.append((startCity1, startCity2))
                    failedTour = True
                self.insertCityIntoTour(cheapestCity, cheapestOrigin, tour)
                if len(tour) == ncities:
                    solutionFound = True
                    break

        end_time = time.time()

        finalList = []
        cost = 0
        for city in tour:
            cost += city.getCity().costTo(city.getForwardConnection().getCity())
            finalList.append(city.getCity())

        results = {}
        results['cost'] = cost
        results['time'] = end_time - startTime
        results['count'] = None
        results['soln'] = TSPSolution(finalList)
        results['max'] = None
        results['total'] = None
        results['pruned'] = None
        return results

    # given new point, add to tour - O(n)
    def insertCityIntoTour(self, cheapestCity, cheapestOrigin, tour):
        # insert into tour at location - O(n)
        tour.insert(tour.index(cheapestOrigin) + 1, cheapestCity)
        # setting forward and backward connections - O(1)
        cheapestCity.setForwardConnection(cheapestOrigin.getForwardConnection())
        cheapestCity.getForwardConnection().setBackwardConnection(cheapestCity)

        cheapestOrigin.setForwardConnection(cheapestCity)
        cheapestCity.setBackwardConnection(cheapestOrigin)

    # finds the closest valid neighbor to add that's not in the tour - O(n^4)
    def findClosestNeighborNotInTour(self, tour, matrix):
        cheapestInsertion = math.inf
        cheapestCity = None
        cheapestOrigin = None

    # for every city neighboring the tour, check if it's connected to the tour-neighbor's
    # forward connection and if so check if it's the cheapest addition so far. If it's the cheapest overall,
    # return it.

        # loop over each city and do operations O(n) * O(n^3) = O(n^4)
        for city in tour: #not real
            # loop over every outgoing node - O(n) * O(n^2) = O(n^3)
            for validForwardCity in city.getOutGoing():
                # check if city in tour - O(n) * O(n) = O(n^2)
                if validForwardCity not in tour:
                    # check if new point connects to forward node then update cheapest stuffs - O(n+1)
                    if city.forwardConnection in validForwardCity.getOutGoing():
                        distToAdd = matrix[city.getIndex()][validForwardCity.getIndex()] + matrix[validForwardCity.getIndex()][city.getForwardConnection().getIndex()]
                        if distToAdd < cheapestInsertion:
                            cheapestInsertion = distToAdd
                            cheapestCity = validForwardCity
                            cheapestOrigin = city
        return cheapestCity, cheapestOrigin, cheapestInsertion

    # make 2d matrix of every distance - O(n^2)
    def makeMatrix(self, cities):

        returnMatrix = []
        for city in cities:
            returnRows = []
            for innerCity in cities:
                returnRows.append(city.costTo(innerCity))
            returnMatrix.append(returnRows)
        return np.array(returnMatrix)

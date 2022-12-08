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
from queue import PriorityQueue
import copy


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
                if currentCity is None:
                    break
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
            # make the queue and get the cities
            # time: O(N^2) for making the matrix, space: O(N^2) for holding the matrix
        queue = PriorityQueue()
        cities = self._scenario.getCities()
        ncities = len(cities)
        first_matrix = self.make_matrix(cities)
        # init the queue and first state
        # time: O(log(N)) for making the state and inserting into queue, space: O(1)
        blank_list = []
        first_state = {"matrix": first_matrix, "path": blank_list,
            "num_visited": ncities-ncities, "curr_node": -1, "curr_child": 0}
        first_state["cost"], first_state["matrix"] = self.calc_cost(
            first_state["matrix"], first_state["curr_node"], first_state["curr_child"])
        queue.put((ncities, first_state["cost"], first_state))

        # start timer
        start_time = time.time()
        # get our initial bssf with random. It doesn't take too long to find a solution
        # time: O(N^2) for random since may need to check all paths per node, space: O(n) for storing cities
        rand_tour = self.defaultRandomTour()
        best_cost = rand_tour['cost']
        best_sol = rand_tour['soln']

        # for everything in the queue, check how good it is and make its children searchable
        # time: O(2^(N)N^(3)), space: O(2^(N)N^(3))
        max_queue_size = 1
        best_counter = 0
        total_created = 1
        total_pruned = 0
        did_change = False
        while not queue.empty() and time.time()-start_time < time_allowance:
                    # update queue size
                    # time: O(1), space: O(1)
            if queue.qsize() > max_queue_size:
                max_queue_size = queue.qsize()
            # check if we've visited all of them
            # time: O(logN), space: O(1) if just accessing memory in the queue
            first_one = queue.get()[2]
            if first_one["num_visited"] >= ncities:
                # print("possible solution", first_one["cost"])
                if first_one["cost"] < best_cost:
                    did_change = True
                    best_cost = first_one["cost"]
                    best_sol = first_one["path"]
                    best_counter += 1
                    # print("new best is", best_cost)

            # prune if the best-case is worse than current best_solution
            # time: O(1), space: O(1)
            if first_one["cost"] > best_cost:
                total_pruned += 1
                continue

            # for each child, make the state, if it's good add to queue
            # O(n) for the children, so total is
            # time: O(n^3), space: O(n^3)
            for row_num in range(ncities):
                # check if we haven't gone there already
                if first_one["matrix"][first_one["curr_child"]][row_num] != math.inf:
                    # update path to include current child
                    # time: O(N), space: O(N)
                    new_path = []
                    for elem in first_one["path"]:
                        new_path.append(elem)
                    new_path.append(row_num)
                    if len(new_path) == 0:
                        print("this is weird...")
                # make child state and find its reduced matrix cost
                # time: O(N^2) for making the matrix, space: O(N^2) for holding the matrix
                    new_state = {"matrix": copy.deepcopy(
                        first_one["matrix"]), "path": new_path, "num_visited": first_one["num_visited"]+1, "curr_node": first_one["curr_child"], "curr_child": row_num}
                    total_created += 1
                    new_state["cost"], new_state["matrix"] = self.calc_cost(
                        new_state["matrix"], new_state["curr_node"], new_state["curr_child"])
                    new_state["cost"] = new_state["cost"] + \
                        first_one["matrix"][first_one["curr_child"]
                                            ][row_num] + first_one["cost"]
                # if it's a good solution, add it to the queue, make sure cost is unique
                # time: O(N), space: O(N)
                    if new_state["cost"] <= best_cost:
                        while any(new_state["cost"] in item for item in queue.queue):
                            new_state["cost"] += 1
                        queue.put(
                            (ncities-new_state["num_visited"], int(new_state["cost"]), new_state))
                    else:
                        total_pruned += 1

        total_pruned += queue.qsize()
        end_time = time.time()
        results = {}
        final_path = []
        # making the final path in TSPSolution form.
        # time: O(N), space: O(N)
        if did_change:
            for stop in best_sol:
                final_path.append(cities[stop])
            final_path = TSPSolution(final_path)
        else:
            final_path = best_sol
        results['cost'] = best_cost
        results['time'] = end_time - start_time
        results['count'] = best_counter
        results['soln'] = final_path
        results['max'] = max_queue_size
        results['total'] = total_created
        results['pruned'] = total_pruned
        # print("max queue size was", max_queue_size)
        return results

    def make_matrix(self, cities):
        # given the starting cities list, make the initial matrix
        # iterate through whole 2d array, so
        # time & space: O(n^2)
        matrix = {}
        for i in range(len(cities)):
            curr_row = {}
            for j in range(len(cities)):
                if i == j:
                    curr_row[j] = math.inf
                else:
                    curr_row[j] = cities[i].costTo(cities[j])
            matrix[i] = curr_row
        return matrix

    def calc_cost(self, new_thing, curr_node, curr_child):
        # given a matrix, finds its reduced cost thing with curr city and curr child
        # time/space: O(n^2)

        # set the row/column/index to infinity for the right places
        # time: O(N), space: O(n^2)
        new_thing = self.inf_the_matrix(
            new_thing, curr_node, curr_child)[1]
        # actually calculate the cost
        min_row = []
        # get the smallest values of the rows and upate them
        # time/space: O(n^2) to iterate through each row
        for row in new_thing.values():
            row = row.values()
            smallest = min(row)
            if smallest == math.inf:
                smallest = 0
            # print("row small", smallest)
            min_row.append(smallest)
        # Update the matrix with the smallest values of the rows
        for row_num in range(len(min_row)):
            for i in range(len(new_thing[0])):
                new_thing[row_num][i] = new_thing[row_num][i] - \
                    min_row[row_num]
        firstset = len(min_row)

        # get the coumn minimum values
        # time/space: O(n^2) to iterate through each column
        min_col = []
        for i in range(len(new_thing[0])):
            all_values = [row[i] for row in new_thing.values()]
            smallest = min(all_values)
            if smallest == math.inf:
                smallest = 0
            # print(smallest)
            min_col.append(smallest)
        # update the column wise stuff
        for col_num in range(len(min_col)):
            for i in range(len(new_thing)):
                new_thing[i][col_num] = new_thing[i][col_num] - \
                    min_col[col_num]
        return (sum(min_row) + sum(min_col), new_thing)

    def inf_the_matrix(self, new_thing, curr_city, curr_child):
        # given a start and end city, make those rows infinity (or return matrix if not valid)
        # time: O(2n), space: O(n^2)
        if curr_city < 0 or curr_child < 0:
            # print("not a thing")
            return (False, new_thing)
        for i in range(len(new_thing[curr_city])):
            new_thing[curr_city][i] = math.inf
        for j in range(len(new_thing)):
            new_thing[j][curr_child] = math.inf
        new_thing[curr_child][curr_city] = math.inf
        return (True, new_thing)

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
            outGoing = matrix[cityNode.getIndex(), :]  # takes whole row
            # add each outgoing to the new dictionary - O(n)
            for i in range(len(outGoing)):
                if outGoing[i] < math.inf:
                    cityNode.addOutGoingNode(nodeDict.get(i))

    def getStarterCities(self, cities, failedPairs, tour, tourDict):
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
        tourDict.update({node1: node1})
        tour.append(node2)
        tourDict.update({node2: node2})
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
            tourDict = {}
            failedTour = False
            startCity1, startCity2 = self.getStarterCities(
                cities, failedPairs, tour, tourDict)
            while not failedTour:
                cheapestCity, cheapestOrigin, cheapestInsertion = self.findClosestNeighborNotInTour(
                    tour, matrix, tourDict)
                if cheapestInsertion == math.inf:
                    failedPairs.append((startCity1, startCity2))
                    failedTour = True
                self.insertCityIntoTour(cheapestCity, cheapestOrigin, tour, tourDict)
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
    def insertCityIntoTour(self, cheapestCity, cheapestOrigin, tour, tourDict):
        # insert into tour at location - O(n)
        tour.insert(tour.index(cheapestOrigin) + 1, cheapestCity)
        tourDict.update({cheapestCity: cheapestCity})
        # setting forward and backward connections - O(1)
        cheapestCity.setForwardConnection(
            cheapestOrigin.getForwardConnection())
        cheapestCity.getForwardConnection().setBackwardConnection(cheapestCity)

        cheapestOrigin.setForwardConnection(cheapestCity)
        cheapestCity.setBackwardConnection(cheapestOrigin)

    # finds the closest valid neighbor to add that's not in the tour - O(n^2)
    def findClosestNeighborNotInTour(self, tour, matrix, tourDict):
        cheapestInsertion = math.inf
        cheapestCity = None
        cheapestOrigin = None

    # for every city neighboring the tour, check if it's connected to the tour-neighbor's
    # forward connection and if so check if it's the cheapest addition so far. If it's the cheapest overall,
    # return it.

        # loop over each city and do operations O(n) * O(n) = O(n^2)
        for city in tour:  # not real
            # loop over every outgoing node - O(n) * O(1) = O(n)
            for validForwardCity in city.getOutGoing():
                # check if city in tour - O(1) * O(1) = O(1)
                if tourDict.get(validForwardCity) is None:
                    # check if new point connects to forward node then update cheapest stuffs - O(1)
                    if validForwardCity.getOutGoing().get(city.forwardConnection) is not None:
                        distToAdd = matrix[city.getIndex()][validForwardCity.getIndex(
                        )] + matrix[validForwardCity.getIndex()][city.getForwardConnection().getIndex()]
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

#!/usr/bin/python
# -*- coding: utf-8 -*-
# Import routines

import numpy as np
import random
from itertools import permutations
from itertools import product

# Defining hyperparameters

m = 5  # number of cities, ranges from 1 ..... m
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger


class CabDriver:

    def __init__(self):
        """initialise your state and define your action space and state space"""

        self.action_space = self.createActionSpace()
        self.state_space = self.createStateSpace()
        self.state_init = self.initializeState()

        # Start the first round

        self.reset()

    def createActionSpace(self):
        actionSpace = [(0,0)]  +  list(permutations([i for i in range(m)], 2))
        return actionSpace

    def createStateSpace(self):
        locations = [i for i in range(m)]
        days = [i for i in range(d)]
        hours = [i for i in range(t)]

        stateSpace = product(locations, days, hours)
        return list(stateSpace)

    def initializeState(self):
        randomInitialState = random.choice(self.state_space)
        return randomInitialState

    def state_encod_arch1(self, state):
        location = state[0]
        day = state[1]
        hour = state[2]

        encodedState = np.zeros((m + d + t, ), dtype=int)
        encodedState[state[0]] = 1
        encodedState[m + state[1]] = 1
        encodedState[m + d + state[2]] = 1

        return encodedState



    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""

        location = state[0]
        requests = self.getRequests(location)

        if requests > 15:
            requests = 15

        possible_actions_index = random.sample(range(1, (m - 1) * m + 1), requests) +[0] # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]



        return (possible_actions_index, actions)

    def getRequests(self, loc):
        lambdaVal = [2, 12, 4, 7, 8]
        requests = np.random.poisson(lambdaVal[loc])
        return requests


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        reward = 0
        cabLocation = state[0]
        day = state[1]
        hour = state[2]
        sourceLocation = action[0]
        targetLocation = action[1]
        #print("CST Locations: ", cabLocation,sourceLocation,targetLocation)

        noRide = False
        if ((sourceLocation == 0) & (targetLocation == 0)):
            noRide = True

        if (noRide):
            reward =  -C

        else :
            timeToSource = (int)(Time_matrix[cabLocation][sourceLocation][hour][day])
            newDay, newHour = self.getNextTime(day, hour, timeToSource)
            #print("Hours: ", day, hour, timeToSource, newDay,newHour)
            timeOfRide = (int)(Time_matrix[sourceLocation][targetLocation][newHour][newDay])
            reward   = (R * timeOfRide) -C * (timeOfRide + timeToSource)

        return reward

    def getNextTime(self, day, hour, timeToSource):
        tempHour = hour + timeToSource
        nextDay = (day + (int)(tempHour/24))%7
        nextHour = tempHour%24
        return nextDay,nextHour

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        cabLocation = state[0]
        day = state[1]
        hour = state[2]
        sourceLocation = action[0]
        targetLocation = action[1]

        reward = self.reward_func(state, action, Time_matrix)

        noRide = False
        if ((sourceLocation == 0) &  (targetLocation == 0)):
            noRide = True

        if (noRide):
            return state, reward, 1

        # time to source plus time to ride
        timeToSource = (int)(Time_matrix[cabLocation][sourceLocation][hour][day])
        newDay, newHour = self.getNextTime(day, hour, timeToSource)
        timeOfRide = (int)(Time_matrix[sourceLocation][targetLocation][newHour][newDay])
        afterRideDay, afterRideHour = self.getNextTime(newDay, newHour, timeOfRide)

        nextState = [targetLocation, afterRideDay, afterRideHour]
        timePassed  = timeToSource + timeOfRide


        return nextState, reward, timePassed

    def reset(self):
        return (self.action_space, self.state_space, self.state_init)
# A General A* Function and its Application to Slide Puzzles
# CS 470/670 at UMass Boston

import numpy as np
from collections import deque
import bisect

example_1_start = np.array([[2, 8, 3],
                            [1, 6, 4],
                            [7, 0, 5]])

example_1_goal = np.array([[1, 2, 3],
                           [8, 0, 4],
                           [7, 6, 5]])

example_2_start = np.array([[2, 6, 4, 8],
                            [5, 11, 3, 12],
                            [7, 0, 1, 15],
                            [10, 9, 13, 14]])

example_2_goal = np.array([[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 0]])


# For a given current state, move, and goal, compute the new state and its h'-score and return them as a pair.
def make_node(state, row_from, col_from, row_to, col_to, goal):
    # Create the new state that results from playing the current move. 
    (height, width) = state.shape
    new_state = np.copy(state)
    new_state[row_to, col_to] = new_state[row_from, col_from]
    new_state[row_from, col_from] = 0

    # Count the mismatched numbers and use this value as the h'-score (estimated number of moves needed to reach the
    # goal).
    mismatch_count = 0
    for i in range(height):
        for j in range(width):
            if new_state[i, j] > 0 and new_state[i, j] != goal[i, j]:
                mismatch_count += 1

    return (new_state, mismatch_count)


# For given current state and goal state, create all states that can be reached from the current state
# (i.e., expand the current node in the search tree) and return a list that contains a pair (state, h'-score)
# for each of these states.   
def slide_expand(state, goal):
    node_list = []
    (height, width) = state.shape
    (empty_row, empty_col) = np.argwhere(state == 0)[0]  # Find the position of the empty tile

    # Based on the position of the empty tile, find all possible moves and add a pair (new_state, h'-score)
    # for each of them.
    if empty_row > 0:
        node_list.append(make_node(state, empty_row - 1, empty_col, empty_row, empty_col, goal))
    if empty_row < height - 1:
        node_list.append(make_node(state, empty_row + 1, empty_col, empty_row, empty_col, goal))
    if empty_col > 0:
        node_list.append(make_node(state, empty_row, empty_col - 1, empty_row, empty_col, goal))
    if empty_col < width - 1:
        node_list.append(make_node(state, empty_row, empty_col + 1, empty_row, empty_col, goal))

    return node_list


# TO DO: Return either the solution as a list of states from start to goal or [] if there is no solution.
def a_star(start, goal, expand):
    # calculate h'-score of start state
    (height, width) = start.shape
    mismatch_count = 0
    # save start state and its h'-score into a list
    open_list = [([start], mismatch_count)]
    # compare start state and goal state then check if goal state is reached
    compare_arrays = np.array_equal(open_list[0][0][-1], goal)
    while not compare_arrays:
        # create all states that can be reached from the current state and return state and h'-score
        node_list = expand(open_list[0][0][-1], goal)
        # check new states and delete same states as ancestors to avoid a search cycle
        node_list_del = []
        for i in range(len(node_list)):
            for j in range(len(open_list[0][0])):
                if np.array_equal(node_list[i][0], open_list[0][0][j]):
                    node_list_del.append(i)
        for i in sorted(node_list_del, reverse=True):
            del node_list[i]
        # if node_list is empty, which means all the new states are same as ancestors, then delete first element of
        # open_list
        if len(node_list) == 0:
            del open_list[0]
        # if node_list is not empty, add all the children states and their f'-scores(current_depth + h'-score) in
        # open_list
        else:
            for i in range(len(node_list)):
                # copy open_list
                ancestor = open_list[0][0].copy()
                ancestor.append(node_list[i][0])
                # save each new state and its ancestors into a list, and calculate its f'-score
                node_list_sub = (ancestor, node_list[i][1] + len(open_list[0][0]))
                open_list.append(node_list_sub)
            # remove the expanded node
            del open_list[0]
            # sort f'-score of the list of open_list
            for i in range(0, len(open_list)):
                for j in range(0, len(open_list) - i - 1):
                    if open_list[j][1] > open_list[j + 1][1]:
                        a = open_list[j]
                        open_list[j] = open_list[j + 1]
                        open_list[j + 1] = a
        # choose the smallest f'-score for next round
        compare_arrays = np.array_equal(open_list[0][0][-1], goal)
        # if open_list is empty, which means that the problem does not have a solution, return empty list
        if len(open_list) == 0:
            return []
    return open_list[0][0]


# Find and print a solution for a given slide puzzle, i.e., the states we need to go through
# in order to get from the start state to the goal state.
def slide_puzzle_solver(start, goal):
    solution = a_star(start, goal, slide_expand_improved)
    if len(solution) == 0:
        print('This puzzle has no solution. Please stop trying to fool me.')
        return

    (height, width) = start.shape
    if height * width >= 10:  # If numbers can have two digits, more space is needed for printing
        digits = 2
    else:
        digits = 1
    horizLine = ('+' + '-' * (digits + 2)) * width + '+'
    for step in range(len(solution)):
        state = solution[step]
        for row in range(height):
            print(horizLine)
            for col in range(width):
                print('| %*d' % (digits, state[row, col]), end=' ')
            print('|')
        print(horizLine)
        if step < len(solution) - 1:
            space = ' ' * (width * (digits + 3) // 2)
            print(space + '|')
            print(space + 'V')


def make_node_improved(state, row_from, col_from, row_to, col_to, goal):
    # Create the new state that results from playing the current move.
    (height, width) = state.shape
    new_state = np.copy(state)
    new_state[row_to, col_to] = new_state[row_from, col_from]
    new_state[row_from, col_from] = 0

    # Count the mismatched numbers and use this value as the h'-score (estimated number of moves needed to reach the
    # goal).
    mismatch_count = 0
    for i in range(height):
        for j in range(width):
            if new_state[i, j] > 0 and new_state[i, j] != goal[i, j]:
                distance = check_distance(new_state, goal, i, j)
                mismatch_count += distance # Adding the distance as the new mismatch value

    return (new_state, mismatch_count)


def check_distance(new_state, goal, i, j):
    """
    Calculates the distance between the tile's current position and the goal position
    """
    result = np.where(goal == new_state[i, j])
    # Calculates the total number of tiles to get to the desired location
    difference = ((i - result[0])**2)**0.5 + ((j - result[1]) ** 2)**0.5
    return difference


# For given current state and goal state, create all states that can be reached from the current state
# (i.e., expand the current node in the search tree) and return a list that contains a pair (state, h'-score)
# for each of these states.
def slide_expand_improved(state, goal):
    node_list = []
    (height, width) = state.shape
    (empty_row, empty_col) = np.argwhere(state == 0)[0]  # Find the position of the empty tile

    # Based on the position of the empty tile, find all possible moves and add a pair (new_state, h'-score)
    # for each of them.
    if empty_row > 0:
        node_list.append(make_node_improved(state, empty_row - 1, empty_col, empty_row, empty_col, goal))
    if empty_row < height - 1:
        node_list.append(make_node_improved(state, empty_row + 1, empty_col, empty_row, empty_col, goal))
    if empty_col > 0:
        node_list.append(make_node_improved(state, empty_row, empty_col - 1, empty_row, empty_col, goal))
    if empty_col < width - 1:
        node_list.append(make_node_improved(state, empty_row, empty_col + 1, empty_row, empty_col, goal))

    return node_list


slide_puzzle_solver(example_1_start, example_1_goal)  # Find solution to example_1
slide_puzzle_solver(example_2_start, example_2_goal)  # Find solution to example_2

# Path_Planning
Implementation of RRT and RTAA* algorithm

Compile main.py to run
RobotPlanner_RTAA.py implement the Realtime Adaptive A* algorithm
RobotPlanner_NRT.py implement the N-directional random exploring tree algorithm

Change the bottom of main.py to test on different test cases

(The following two function are exclusive, if you plan to run one algorithm, make sure to comment the part of the other one)
(You are welcomed to test on different hyperparameters)

1. RTAA*
To test the RRTA* algorithm, uncomment the first part (line 72-75) in function runtest, also uncomment line 98 in while loop.
There are three hyper-parameters in this algorithm:
	a. step_size 	--	collaborate with the minimum obstacle width to determine the discretization resolution of the map
	b. expand_size 	--	number of nodes to expand for one timestamp, make sure not to set this too large, or the algorithm will exceed 						2s planning time constraint and your machine will run out of memory
	c. epsilon		--	The weight factor of heuristic value  

2. NRT
To test the NRT algorithm, uncomment the second part (line 78-81) in function runtest, also uncomment line 99 in while loop.
There are three hyper-parameters in this algorithm:
	a. n 			--	number of extra trees to expand simutaneously besides the start and end tree, the minimum value of n is 0 (							bi-directional). Default value is 100, if too small, the planning time will become untolerable for some 							complicated cases (e.g. maze)
	b. rewiring		--	Whether to do the rewiring step
	c. r 			--	The rewiring threshold, only consider the node within distance r of the current node, if set too large, the 						algorithm will become slow

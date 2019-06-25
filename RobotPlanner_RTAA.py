import numpy as np
import heapq as pq


class RobotPlanner_RTAA:
    __slots__ = ['boundary', 'blocks', 'MAP', 'heuristic', 'path']

    def __init__(self, boundary, blocks, minstep):
        self.boundary = boundary
        self.blocks = blocks
        # Choose map resolution based on block width
        for k in range(self.blocks.shape[0]):
            minstep = min(minstep,  abs(blocks[k,0]-blocks[k,3]),
                                    abs(blocks[k,1]-blocks[k,4]),
                                    abs(blocks[k,2]-blocks[k,5]))
        self.MAP = {}
        self.MAP['res'] = minstep # resolution in meters
        self.MAP['xmin'] = boundary[0,0]  # meters
        self.MAP['ymin'] = boundary[0,1]
        self.MAP['zmin'] = boundary[0,2]
        self.MAP['xmax'] = boundary[0,3]
        self.MAP['ymax'] = boundary[0,4]
        self.MAP['zmax'] = boundary[0,5]
        self.MAP['sizex'] = int(np.floor((self.MAP['xmax'] - self.MAP['xmin']) / self.MAP['res'] + 1)) 
        self.MAP['sizey'] = int(np.floor((self.MAP['ymax'] - self.MAP['ymin']) / self.MAP['res'] + 1))
        self.MAP['sizez'] = int(np.floor((self.MAP['zmax'] - self.MAP['zmin']) / self.MAP['res'] + 1))
        self.heuristic = np.ones((self.MAP['sizex'], self.MAP['sizey'], self.MAP['sizez']), dtype=np.float32)*-1
        self.path = []

    # Convert physical coordinates to grid index
    def physical_to_grid(self, coor):
        coor = np.floor((coor - np.array([self.MAP['xmin'],self.MAP['ymin'],self.MAP['zmin']])) / self.MAP['res'] ).astype(np.int16)
        return coor

    # Convert grid index to physical coordinates
    def grid_to_physical(self, coor):
        coor = coor*self.MAP['res'] + np.array([self.MAP['xmin'],self.MAP['ymin'],self.MAP['zmin']])
        return coor

    def get_heuristic(self, start_coor, goal_coor):
        #return np.linalg.norm(start_coor-goal_coor)    # Euclidean heuristic
        return np.sum(abs(start_coor-goal_coor))        # Manhattan heuristic

    # Real-Time Adaptive A* Algorithm
    def rtaa_plan(self, start, goal, expand_size, epsilon):
        # Finish pre-determined path first
        if (self.path):
            return self.path.pop()

        # Get grid coordinates of start and goal 
        start_coor = self.physical_to_grid(start) 
        goal_coor = self.physical_to_grid(goal) 

        numofdirs = 26
        [dX,dY,dZ] = np.meshgrid([-1,0,1],[-1,0,1],[-1,0,1])
        dR = np.vstack((dX.flatten(),dY.flatten(),dZ.flatten()))
        dR = np.delete(dR,13,axis=1)

        OPEN = [] # Implemented as priority queue
        if (self.heuristic[tuple(start_coor)] == -1):
            self.heuristic[tuple(start_coor)] = self.get_heuristic(start_coor, goal_coor)
        pq.heappush(OPEN,(epsilon*self.heuristic[tuple(start_coor)], tuple(start_coor)))
        CLOSED = {}
        gvalue = np.ones((self.MAP['sizex'], self.MAP['sizey'], self.MAP['sizez']), dtype=np.float32)*1000000
        gvalue[tuple(start_coor)] = 0
        parent = {}

        flag = 0 # Whether target node is found
        for i in range(expand_size):
            curr_node = pq.heappop(OPEN)
            while (curr_node[1] in CLOSED):
                curr_node = pq.heappop(OPEN)
            CLOSED[curr_node[1]] = curr_node[0]
            curr_node = curr_node[1]
            if (np.array_equal(curr_node,goal_coor)):
                flag = 1
                break

            for k in range(numofdirs):
                newrp_coor = curr_node + dR[:,k].flatten()
                newrp = self.grid_to_physical(newrp_coor)

                # Check if this direction is closed
                if (tuple(newrp_coor) in CLOSED):
                    continue
                # Check if this direction is valid
                if( newrp[0] <= self.boundary[0,0] or newrp[0] >= self.boundary[0,3] or \
                    newrp[1] <= self.boundary[0,1] or newrp[1] >= self.boundary[0,4] or \
                    newrp[2] <= self.boundary[0,2] or newrp[2] >= self.boundary[0,5] ):
                    continue
                valid = True
                for k in range(self.blocks.shape[0]):
                    if( newrp[0] >= self.blocks[k,0] and newrp[0] <= self.blocks[k,3] and\
                        newrp[1] >= self.blocks[k,1] and newrp[1] <= self.blocks[k,4] and\
                        newrp[2] >= self.blocks[k,2] and newrp[2] <= self.blocks[k,5] ):
                        valid = False
                        break
                if not valid:
                    continue

                # If valid and gj > gi+cij, insert j into OPEN list
                if (gvalue[tuple(newrp_coor)] > gvalue[tuple(curr_node)]+self.get_heuristic(curr_node,newrp_coor)):
                    gvalue[tuple(newrp_coor)] = gvalue[tuple(curr_node)]+self.get_heuristic(curr_node,newrp_coor)
                    if (self.heuristic[tuple(newrp_coor)] == -1):
                        self.heuristic[tuple(newrp_coor)] = self.get_heuristic(newrp_coor, goal_coor) 
                    pq.heappush(OPEN, (gvalue[tuple(newrp_coor)]+epsilon*self.heuristic[tuple(newrp_coor)], tuple(newrp_coor)))
                    parent[tuple(newrp_coor)] = tuple(curr_node)

        # If goal has been reached
        if (flag == 1):
            self.path.append(goal)
            while (not np.array_equal(parent[tuple(goal_coor)], start_coor)):
                self.path.append(self.grid_to_physical(np.array(goal_coor)))
                goal_coor = parent[tuple(goal_coor)]
            return self.grid_to_physical(np.array(goal_coor))

        # Update heuristic value
        curr_node = pq.heappop(OPEN)
        while (curr_node[1] in CLOSED):
            curr_node = pq.heappop(OPEN)
        target_node = curr_node[1]
        f_star = curr_node[0]

        for node in CLOSED: 
            self.heuristic[node] = f_star - gvalue[node]

        while (not np.array_equal(parent[target_node], start_coor)):
            self.path.append(self.grid_to_physical(np.array(target_node)))
            target_node = parent[target_node]

        return self.grid_to_physical(np.array(target_node))
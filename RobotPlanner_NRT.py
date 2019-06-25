import numpy as np
import heapq as pq
import time

class RobotPlanner_NRT:
    __slots__ = ['boundary', 'blocks', 'path', 'numofnodes', 'minstep', 'nodedict', 'nodedict2', 'treelist', 'adj_mat', 'find', 'rewiring', 'r']

    def __init__(self, boundary, blocks, n, start, goal, rewiring, r):
        self.boundary = boundary
        self.blocks = blocks
        self.path = []  # the optimal path
        self.numofnodes = 0
        # Choose minimum step_size to check collision
        self.minstep = 0.1
        for k in range(self.blocks.shape[0]):
            self.minstep = min(self.minstep,abs(blocks[k,0]-blocks[k,3]),
                                            abs(blocks[k,1]-blocks[k,4]),
                                            abs(blocks[k,2]-blocks[k,5]))
        # Initialize node dictionary & tree list & adjacency matrix
        self.nodedict = {}
        self.nodedict2 = {}
        self.add_node(start)
        self.add_node(goal)
        self.treelist = []
        self.treelist.append(np.array([start]))
        self.treelist.append(np.array([goal]))
        self.adj_mat = np.zeros((2,2))
        for i in range(n):
            point = self.sample_point()
            self.add_node(point[0])
            self.treelist.append(point)
            self.update_adj(i+2,i+2,0)
        self.find = False
        self.rewiring = rewiring
        self.r = r

    # Find all points on a line segments using minimum step_size
    def discretize_path(self, start, end, minstep):
        num_of_points = np.ceil(np.linalg.norm(start-end)/minstep)+3
        return np.round(start + np.linspace(0, 1, num_of_points)[..., np.newaxis] * (end - start),5)

    def tic(self):
        return time.time()

    # Randomly sample a point in free space
    def sample_point(self):
        sample = np.round(np.array([[np.random.uniform(self.boundary[0,0],self.boundary[0,3],None),
                                     np.random.uniform(self.boundary[0,1],self.boundary[0,4],None),
                                     np.random.uniform(self.boundary[0,2],self.boundary[0,5],None)]]),5)
        while(not self.check_valid(sample[0])):
            sample = np.round(np.array([[np.random.uniform(self.boundary[0,0],self.boundary[0,3],None),
                                         np.random.uniform(self.boundary[0,1],self.boundary[0,4],None),
                                         np.random.uniform(self.boundary[0,2],self.boundary[0,5],None)]]),5)
        return sample

    # Return true if the point is in free space
    def check_valid(self, point):
        for k in range(self.blocks.shape[0]):
            if( point[0] >= self.blocks[k,0] and point[0] <= self.blocks[k,3] and\
                point[1] >= self.blocks[k,1] and point[1] <= self.blocks[k,4] and\
                point[2] >= self.blocks[k,2] and point[2] <= self.blocks[k,5] ):
                return False
        return True

    # Add current node to hash table
    def add_node(self, point):
        self.nodedict[tuple(point)] = self.numofnodes
        self.nodedict2[self.numofnodes] = tuple(point)
        self.numofnodes += 1
        return

    # Update the adjacency matrix
    def update_adj(self, i, j, cost):
        curr_size = np.shape(self.adj_mat)[0]
        temp = np.zeros((curr_size+1,curr_size+1))
        temp[0:curr_size,0:curr_size] = self.adj_mat
        self.adj_mat = temp
        self.adj_mat[i,j] = cost
        self.adj_mat[j,i] = cost
        return

    # Calculate the euclidean distance between two points
    def get_distance(self, start, end):
        if (np.shape(np.shape(start))[0]==1):
            return np.linalg.norm(start-end)
        return np.linalg.norm(start-end,axis=1)

    # Rewire the tree
    def rewire_node(self, treeindex, curr_node_index):
        curr_node = np.array(self.nodedict2[curr_node_index])
        near_point = self.treelist[treeindex][np.where(self.get_distance(self.treelist[treeindex], curr_node) < self.r)]
        for i in range(np.shape(near_point)[0]):  
            flag = True
            start_node = near_point[i]
            start_node_index = self.nodedict[tuple(start_node)]
            all_points = self.discretize_path(start_node, curr_node, self.minstep)
            for j in range(np.shape(all_points)[0]):
                if (not self.check_valid(all_points[j])):
                    flag = False
                    break
            if (flag):
                cost = self.get_distance(curr_node,start_node)
                self.adj_mat[start_node_index,curr_node_index] = cost
                self.adj_mat[curr_node_index,start_node_index] = cost

    # Tree has been constructed, use dijkstra to find the optimal path
    def dijkstra(self):
        visited = np.zeros(self.numofnodes)
        currcost = np.full((self.numofnodes), np.inf)
        ancestor = np.full((self.numofnodes), np.inf, dtype=np.int32)
        OPEN = []
        pq.heappush(OPEN,(0, 0))
        currcost[0] = 0;
        ancestor[0] = 0
        while(np.size(OPEN)!=0):
            curr = pq.heappop(OPEN)
            if (visited[curr[1]]==1):
                continue
            visited[curr[1]] = 1;
            for i in range(self.numofnodes):
                if (self.adj_mat[curr[1],i]==0):
                    continue
                if (self.adj_mat[curr[1],i]+curr[0] < currcost[i]):
                    currcost[i] = self.adj_mat[curr[1],i]+curr[0]
                    ancestor[i] = curr[1]
                    pq.heappush(OPEN,(self.adj_mat[curr[1],i]+curr[0],i))
            # Early break if node has been found
            if (curr[1]==1):
                break
        
        # Trace back the optimal path
        path = np.array([],dtype = np.int32)
        currnode = 1
        while(currnode!=0):
            path = np.append(path, currnode)
            currnode = ancestor[currnode]

        path = np.append(path, currnode)

        for i in range(len(path)-1):
            self.path = self.path + self.discretize_path(np.array(self.nodedict2[path[i]]),np.array(self.nodedict2[path[i+1]]),0.9).tolist()
        return


    # N-Directional Rapidly Exploring Random Tree Algorithm
    def nrt_plan(self, start, goal):
        t0 = self.tic()

        # Finish pre-determined path first
        if (self.path):
            return np.array(self.path.pop())

        if (self.find):
            self.dijkstra()
            return start

        while True:
            # If planning time is about to exceed 2s, let the robot stay where it was
            if ((self.tic()-t0)/2.0>0.95):
                return start

            # Randomly sample a point in free space
            curr_point = self.sample_point()
            # Record how many tree segments can this point be connected to 
            connected_tree = []
            connected_point = []
            for i in range(len(self.treelist)):
                closest_point = self.treelist[i][np.argmin(self.get_distance(self.treelist[i], curr_point[0]))]
                all_points = self.discretize_path(closest_point, curr_point[0], self.minstep)
                for j in range(np.shape(all_points)[0]):
                    last_valid = j
                    if (not self.check_valid(all_points[j])):
                        last_valid -= 1
                        break
                if (last_valid == 0):
                    continue
                # the point can be connected to this tree
                if (last_valid == np.shape(all_points)[0]-1):
                    connected_tree.append(i)
                    connected_point.append(closest_point)
                    continue
                # Can not directly connect to this tree, expand the tree to the furthest possible point
                self.add_node(all_points[last_valid])
                self.update_adj(self.nodedict[tuple(closest_point)],self.numofnodes-1,self.get_distance(closest_point,all_points[last_valid]))
                self.treelist[i] = np.vstack((self.treelist[i],all_points[last_valid]))
                # Rewire this node to all other nodes in this tree
                if (self.rewiring):
                    self.rewire_node(i,self.numofnodes-1)

            
            # Connect this point to all possible trees
            if (len(connected_tree)==0):
                continue
            else:
                # Add this node to node list
                self.add_node(curr_point[0])
                # Update adjacency matrix
                self.update_adj(self.nodedict[tuple(connected_point[0])],self.numofnodes-1,self.get_distance(connected_point[0],curr_point[0]))
                for i in range(1,len(connected_point)):
                    m = self.nodedict[tuple(connected_point[i])]
                    n = self.numofnodes-1
                    curr_distance = self.get_distance(connected_point[i],curr_point[0])
                    self.adj_mat[m,n] = curr_distance
                    self.adj_mat[n,m] = curr_distance
                # Update tree list
                self.treelist[connected_tree[0]] = np.vstack((self.treelist[connected_tree[0]],curr_point))
                if (len(connected_tree) == 1):
                    if (self.rewiring):
                        self.rewire_node(connected_tree[0],self.numofnodes-1)
                    continue
                # If path found, run dijkstra to find the optimal path
                if (connected_tree[0]==0 and connected_tree[1]==1):
                    if (self.rewiring):
                        self.rewire_node(connected_tree[0],self.numofnodes-1)
                        self.rewire_node(connected_tree[1],self.numofnodes-1)
                    self.find = True
                    return start
                else:
                    tree_index = connected_tree[0]
                    for j in range(1,len(connected_tree)):
                        self.treelist[tree_index] = np.vstack((self.treelist[tree_index],self.treelist[connected_tree[j]]))
                    del connected_tree[0]
                    for j in sorted(connected_tree, reverse=True):
                        del self.treelist[j]

                if (self.rewiring):
                    self.rewire_node(tree_index,self.numofnodes-1)
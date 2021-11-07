# Databricks notebook source
# MAGIC %md
# MAGIC # PLEASE CLONE THIS NOTEBOOK INTO YOUR PERSONAL FOLDER
# MAGIC # DO NOT RUN CODE IN THE SHARED FOLDER

# COMMAND ----------

# MAGIC %md # HW 5 - Page Rank
# MAGIC __`MIDS w261: Machine Learning at Scale | UC Berkeley School of Information | Fall 2018`__
# MAGIC 
# MAGIC In Weeks 8 and 9 you discussed key concepts related to graph based algorithms and implemented SSSP.   
# MAGIC In this final homework assignment you'll implement distributed PageRank using some data from Wikipedia.
# MAGIC By the end of this homework you should be able to:  
# MAGIC * ... __compare/contrast__ adjacency matrices and lists as representations of graphs for parallel computation.
# MAGIC * ... __explain__ the goal of the PageRank algorithm using the concept of an infinite Random Walk.
# MAGIC * ... __define__ a Markov chain including the conditions underwhich it will converge.
# MAGIC * ... __identify__ what modifications must be made to the web graph in order to leverage Markov Chains.
# MAGIC * ... __implement__ distributed PageRank in Spark.
# MAGIC 
# MAGIC __Please refer to the `README` for homework submission instructions and additional resources.__ 

# COMMAND ----------

# MAGIC %md # Notebook Set-Up
# MAGIC Before starting your homework run the following cells to confirm your setup.   

# COMMAND ----------

# imports
import re
import ast
import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt

# COMMAND ----------

# RUN THIS CELL AS IS. 
tot = 0
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
for item in dbutils.fs.ls(DATA_PATH):
  tot = tot+item.size
tot
# ~4.7GB

# COMMAND ----------

# RUN THIS CELL AS IS. You should see all-pages-indexed-in.txt, all-pages-indexed-out.txt and indices.txt in the results. If you do not see these, please let an Instructor or TA know.
display(dbutils.fs.ls(DATA_PATH))

# COMMAND ----------

sc = spark.sparkContext
spark

# COMMAND ----------

# MAGIC %md # Question 1: Distributed Graph Processing
# MAGIC Chapter 5 from Lin & Dyer gave you a high level introduction to graph algorithms and concerns that come up when trying to perform distributed computations over them. The questions below are designed to make sure you captured the key points from this reading and your async lectures. 
# MAGIC 
# MAGIC ### Q1 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ Give an example of a dataset that would be appropriate to represent as a graph. What are the nodes/edges in this dataset? Is the graph you describe 'directed' or 'undirected'? What would the average "in-degree" of a node mean in the context of your example? 
# MAGIC 
# MAGIC * __b) short response:__ Other than their size/scale, what makes graphs uniquely challenging to work with in the map-reduce paradigm? *(__HINT__: Do not respond in terms of any specific algorithm. Think in terms of the nature of the graph datastructure itself).*
# MAGIC 
# MAGIC * __c) short response:__ Briefly describe Dijskra's algorithm (goal/approach). What specific design component makes this approach hard to parallelize?
# MAGIC 
# MAGIC * __d) short response:__ How does parallel breadth-first-search get around the problem that you identified in part `c`? At what expense?

# COMMAND ----------

# MAGIC %md ### Q1 Student Answers:
# MAGIC > __a)__ **Give an example of a dataset that would be appropriate to represent as a graph. What are the nodes/edges in this dataset? Is the graph you describe 'directed' or 'undirected'? What would the average "in-degree" of a node mean in the context of your example?**
# MAGIC >   
# MAGIC > A social network is an easy one. The nodes would be people and the edges would be whether or not they are connected.   
# MAGIC >
# MAGIC > Is the graph directed? They're not usually considered directed or to even have weights, but they could be. This is less true of a social network like Facebook, but it could be true of LinkedIn. For example, I have ~1000 LinkedIn connections. Some of them are people that reach out to me about as often as I reach out to them and are about as influential as I am. Our relationship could be considered undirected (or bidirectional) and of roughly equal weights. On the other hand, I know a handful of CEOs and CFOs. They have way more influence (e.g. they're connections are professionally more valuable) and they reach out to me much less often than I reach out to them. These connections could be considered to have much greater weights and something akin to being directional. 
# MAGIC >  
# MAGIC > **What would the average "in-degree" of a node mean in the context of your example?**
# MAGIC >  
# MAGIC > In the context of my example, using the undirected case, "in-degree" just means "degree". The average "in-degree" in a social network like LinkedIn would be the average number of connections each member of the network had. 
# MAGIC 
# MAGIC 
# MAGIC > __b)__ **Other than their size/scale, what makes graphs uniquely challenging to work with in the map-reduce paradigm?**   
# MAGIC >  
# MAGIC > It's challenging to write map-reduce graph algorithms that don't have state dependencies across nodes. For example, one worker working on finding the shortest path towards a particular node might find a path of length `n` while another might find one with a length of `n-1`. Writing algorithms that deliver the shortest path in cases like this while working across different nodes is challenging. 
# MAGIC 
# MAGIC 
# MAGIC > __c)__ **Briefly describe Dijkstra's algorithm (goal/approach). What specific design component makes this approach hard to parallelize?**
# MAGIC >  
# MAGIC >  Dijkstra's algorithm is an algorithm that can find the shortest path in a weighted directed graph. Its approach is to iteratively select the node with the lowest current distance from the priority queue and expand *that* node's adjacency list to see if those nodes can be reached with a path with a shorter distance. The design component that makes this difficult to parallelize is the need for a global priority queue, something generally inconsistent with independent workers.
# MAGIC 
# MAGIC 
# MAGIC > __d)__ **How does parallel breadth-first-search get around the problem that you identified in part `c`? At what expense?**
# MAGIC >  
# MAGIC >  Parallel breadth-first-search gets around a global priority queue by performing splitting the search frontier across multiple mappers, allowing the work to be parallelized. This allows multiple paths to the same node to be found by different mappers. The reducers reconcile these by selecting the one with the shortest path. The expense is potentially a lot of extra, unneccessary work is done. Lin & Dyer characterize this unnecessary work as work done within the search frontier (redundant, repeated work). The productive work is done *at* the search frontier, where potentially new, shorter paths are found. 

# COMMAND ----------

# MAGIC %md # Question 2: Representing Graphs 
# MAGIC 
# MAGIC In class you saw examples of adjacency matrix and adjacency list representations of graphs. These data structures were probably familiar from HW3, though we hadn't before talked about them in the context of graphs. In this question we'll discuss some of the tradeoffs associated with these representations. __`NOTE:`__ We'll use the graph from Figure 5.1 in Lin & Dyer as a toy example. For convenience in the code below we'll label the nodes `A`, `B`, `C`, `D`, and `E` instead of $n_1$, $n_2$, etc but otherwise you should be able to follow along & check our answers against those in the text.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://github.com/kyleiwaniec/w261_assets/blob/master/images/HW5/Lin-Dyer-graph-Q1.png?raw=true" width=50%>
# MAGIC 
# MAGIC ### Q2 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ Relatively speaking, is the graph you described in Figure 5.1 in Lin & Dyer "sparse" or "dense"?  Explain how sparsity/density impacts the adjacency matrix and adjacency list representations of a graph.
# MAGIC 
# MAGIC * __b) short response:__ Run the provided code to create and plot our toy graph. Is this graph directed or undirected? Explain how the adjacency matrices for directed graphs will differ from those of undirected graphs.
# MAGIC 
# MAGIC * __c) code:__ Fill in the missing code to complete the function `get_adj_matr()`.
# MAGIC 
# MAGIC * __d) code:__ Fill in the missing code to complete the function `get_adj_list()`.

# COMMAND ----------

# MAGIC %md ### Q2 Student Answers:
# MAGIC > __a)__ **Relatively speaking, is the graph you described in Figure 5.1 in Lin & Dyer "sparse" or "dense"?  Explain how sparsity/density impacts the adjacency matrix and adjacency list representations of a graph.**   
# MAGIC >   
# MAGIC > With no self loops, there are 5^2 - 5 = 20 potential connections in this directed graph. There are 9 actual connections. With actual connections representing less than half the potential ones, I'd characterize this as a "sparse" graph. 
# MAGIC >  
# MAGIC > With a sparse graph, an adjacency matrix devotes lots of space to representing connections that don't exist. An adjacency list will take up much less memory/space. With a dense graph, an adjacency lists and adjacency matrices will have similar memory and storage requirements. In the limit, where every possible connection exists, adjacency lists and adjacency matrices will hold the same number of representations in memory. 
# MAGIC 
# MAGIC 
# MAGIC > __b)__ **Run the provided code to create and plot our toy graph. Is this graph directed or undirected? Explain how the adjacency matrices for directed graphs will differ from those of undirected graphs.**   
# MAGIC >   
# MAGIC > This is a directed graph. The adjacency matrix for an undirected graph is symmetric. If there's a connection between node A and B, that implies a connection between node B and A and vice versa. That symmetry need not exist in a directed graph's adjacency matrix. It's quite possible for there to be a connection between node A and B with no corresponding connection between node B and A. (Direction matters with directed graphs.)

# COMMAND ----------

# part a - a graph is just a list of nodes and edges (RUN THIS CELL AS IS)
TOY_GRAPH = {'nodes':['A', 'B', 'C', 'D', 'E'],
             'edges':[('A', 'B'), ('A', 'D'), ('B', 'C'), ('B', 'E'), ('C', 'D'), 
                      ('D', 'E'), ('E', 'A'),('E', 'B'), ('E', 'C')]}

# COMMAND ----------

# part a - simple visualization of our toy graph using nx (RUN THIS CELL AS IS)
G = nx.DiGraph()
G.add_nodes_from(TOY_GRAPH['nodes'])
G.add_edges_from(TOY_GRAPH['edges'])
display(nx.draw(G, pos=nx.circular_layout(G), with_labels=True, alpha = 0.5))

# COMMAND ----------

# part c - adjacency matrix function
def get_adj_matr(graph):
    """
    Function to create an adjacency matrix representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        pd.DataFrame with entry i,j representing an edge from node i to node j
    """
    n = len(graph['nodes'])
    adj_matr = pd.DataFrame(0, columns = graph['nodes'], index = graph['nodes'])
    ############### YOUR CODE HERE ##################
    for edge in graph['edges']:
      adj_matr.loc[edge[0], edge[1]] = 1
    
    ############### (END) YOUR CODE #################
    return adj_matr

# COMMAND ----------

# part c - take a look (RUN THIS CELL AS IS)
TOY_ADJ_MATR = get_adj_matr(TOY_GRAPH)
print(TOY_ADJ_MATR)

# COMMAND ----------

TOY_GRAPH

# COMMAND ----------

# part d - adjacency list function
def get_adj_list(graph):
    """
    Function to create an adjacency list representation of a graph.
    arg:
        graph - (dict) of 'nodes' : [], 'edges' : []
    returns:
        dictionary of the form {node : [list of edges]}
    """
    adj_list = {node: [] for node in graph['nodes']}
    ############### YOUR CODE HERE ##################
    
    for edge in graph['edges']:
      adj_list[edge[0]].append(edge[1])
    
    ############### (END) YOUR CODE #################
    return adj_list

# COMMAND ----------

# part d - take a look (RUN THIS CELL AS IS)
TOY_ADJ_LIST = get_adj_list(TOY_GRAPH)
print(TOY_ADJ_LIST)

# COMMAND ----------

# MAGIC %md # Question 3: Markov Chains and Random Walks
# MAGIC 
# MAGIC As you know from your readings and in class discussions, the PageRank algorithm takes advantage of the machinery of Markov Chains to compute the relative importance of a webpage using the hyperlink structure of the web (we'll refer to this as the 'web-graph'). A Markov Chain is a discrete-time stochastic process. The stochastic matrix has a principal left eigen vector corresponding to its largest eigen value which is one. A Markov chain's probability distribution over its states may be viewed as a probability vector. This steady state probability for a state is the PageRank of the corresponding webpage. In this question we'll briefly discuss a few concepts that are key to understanding the math behind PageRank. 
# MAGIC 
# MAGIC ### Q3 Tasks:
# MAGIC 
# MAGIC * __a) short response:__ It is common to explain PageRank using the analogy of a web surfer who clicks on links at random ad infinitum. In the context of this hypothetical infinite random walk, what does the PageRank metric measure/represent?
# MAGIC 
# MAGIC * __b) short response:__ What is the "Markov Property" and what does it mean in the context of PageRank?
# MAGIC 
# MAGIC * __c) short response:__ A Markov chain consists of \\$n$\\ states plus an $n\times n$ transition probability matrix. In the context of PageRank & a random walk over the WebGraph what are the $n$ states? what implications does this have about the size of the transition matrix?
# MAGIC 
# MAGIC * __d) code + short response:__ What is a "right stochastic matrix"? Fill in the code below to compute the transition matrix for the toy graph from question 2. [__`HINT:`__ _It should be right stochastic. Using numpy this calculation can be done in one line of code._]
# MAGIC 
# MAGIC * __e) code + short response:__ To compute the stable state distribution (i.e. PageRank) of a "nice" graph we can apply the power iteration method - repeatedly multiplying the transition matrix by itself, until the values no longer change. Apply this strategy to your transition matrix from `part d` to find the PageRank for each of the pages in your toy graph. Your code should print the results of each iteration. How many iterations does it take to converge? Which node is most 'central' (i.e. highest ranked)? Does this match your intuition? 
# MAGIC     * __`NOTE 1:`__ _this is a naive approach, we'll unpack what it means to be "nice" in the next question_.
# MAGIC     * __`NOTE 2:`__ _no need to implement a stopping criteria, visual inspection should suffice_.

# COMMAND ----------

# MAGIC %md ### Q3 Student Answers:
# MAGIC > __a)__ **It is common to explain PageRank using the analogy of a web surfer who clicks on links at random ad infinitum. In the context of this hypothetical infinite random walk, what does the PageRank metric measure/represent?**  
# MAGIC >  
# MAGIC > The PageRank measure represents the proportion of the time the proverbial web surfer reaches each node. 
# MAGIC 
# MAGIC > __b)__ **What is the "Markov Property" and what does it mean in the context of PageRank?**   
# MAGIC >   
# MAGIC > Memorylessness with regard to state changes. The next state only depends on the current state and not the sequence of events that preceded it. In the context of PageRank, it means given a particular node, the probability of the next node in the random walk of our proverbial web surfer only depends on the the properties of that particular node and not in any way on the path they took to get to that node. (In this case, the probabilities of moving to one of the nodes linked to from this particular node.) 
# MAGIC 
# MAGIC 
# MAGIC > __c)__ **A Markov chain consists of $n$ states plus an $n\times n$ transition probability matrix. In the context of PageRank & a random walk over the WebGraph what are the $n$ states? what implications does this have about the size of the transition matrix?**  
# MAGIC >   
# MAGIC > The n states are just the n web pages/nodes in the graph. n states implies n x n possible transitions (i.e. from each node to every other node, including itself), which means the transition matrix has a size of n x n = n^2.
# MAGIC 
# MAGIC 
# MAGIC > __d)__ **What is a "right stochastic matrix"? Fill in the code below to compute the transition matrix for the toy graph from question 2.**  
# MAGIC >  
# MAGIC > From the aync, it is: A right stochastic matrix is a square matrix of nonnegative real numbers, with each row summing to 1. If the row vector of node probabilities is on the left, when the row vector is multiplied by the right stochastic matrix, its weights are updated to a new probability distribution. 
# MAGIC 
# MAGIC > __e)__ **To compute the stable state distribution (i.e. PageRank) of a "nice" graph we can apply the power iteration method - repeatedly multiplying the transition matrix by itself, until the values no longer change. Apply this strategy to your transition matrix from `part d` to find the PageRank for each of the pages in your toy graph. Your code should print the results of each iteration. How many iterations does it take to converge? Which node is most 'central' (i.e. highest ranked)? Does this match your intuition?** 
# MAGIC     * __`NOTE 1:`__ _this is a naive approach, we'll unpack what it means to be "nice" in the next question_.
# MAGIC     * __`NOTE 2:`__ _no need to implement a stopping criteria, visual inspection should suffice_.**  
# MAGIC >  
# MAGIC > It seems to take 50 iterations to converge to the steady state result. However, it gets very close to this figure quite a bit earlier. The highest rank node is E. Visually, the way the graph is drawn with E at the center certainly makes it *seem* central, but I'm not sure what my intuition was. The rows tell you how many ways out of each node there are, but I'm not sure how much that makes a node more central. The columns tell you how many ways IN to each node there are, but 4 different nodes have 2 ways in, not just e. I think the right way to think of this is to go a step further back and see how many ways there are into a node which, in turn, can feed into each node. Obviously, you can keep going backwards with this. Given all this, I'm not sure how much I'd trust my intuition.
# MAGIC     
# MAGIC     
# MAGIC     
# MAGIC     

# COMMAND ----------

# part d - recall what the adjacency matrix looked like (RUN THIS CELL AS IS)
TOY_ADJ_MATR

# COMMAND ----------

# part d - use TOY_ADJ_MATR to create a right stochastic transition matrix for this graph
################ YOUR CODE HERE #################
#transition_matrix = None # replace with your code

transition_matrix = TOY_ADJ_MATR.to_numpy()
transition_matrix = transition_matrix/transition_matrix.sum(axis = 1)[:, None]

################ (END) YOUR CODE #################
print(transition_matrix)

# COMMAND ----------

# part e - compute the steady state using the transition matrix 
def power_iteration(xInit, tMatrix, nIter, verbose = True):
    """
    Function to perform the specified number of power iteration steps to 
    compute the steady state probability distribution for the given
    transition matrix.
    
    Args:
        xInit     - (n x 1 array) representing inial state
        tMatrix  - (n x n array) transition probabilities
        nIter     - (int) number of iterations
    Returns:
        state_vector - (n x 1 array) representing probability 
                        distribution over states after nSteps.
    
    NOTE: if the 'verbose' flag is on, your function should print the step
    number and the current matrix at each iteration.
    """
    state_vector = xInit
    ################ YOUR CODE HERE #################

    for i in range(nIter):
      
      state_vector = np.matmul(state_vector, tMatrix)
      if verbose == True:
        print(f"Step number: {i}, State vector: {state_vector}, Sum of Weights: {np.sum(state_vector)}")
    
    
    
    ################ (END) YOUR CODE #################
    return state_vector

# COMMAND ----------

# part e - run 10 steps of the power_iteration (RUN THIS CELL AS IS)
xInit = np.array([1.0, 0, 0, 0, 0]) # note that this initial state will not affect the convergence states
states = power_iteration(xInit, transition_matrix, 50, verbose = True)

# COMMAND ----------

# MAGIC %md __`Expected Output for part e:`__  
# MAGIC >Steady State Probabilities:
# MAGIC ```
# MAGIC Node A: 0.10526316  
# MAGIC Node B: 0.15789474  
# MAGIC Node C: 0.18421053  
# MAGIC Node D: 0.23684211  
# MAGIC Node E: 0.31578947  
# MAGIC ```

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # Question 4: Page Rank Theory
# MAGIC 
# MAGIC Seems easy right? Unfortunately applying this power iteration method directly to the web-graph actually runs into a few problems. In this question we'll tease apart what we meant by a 'nice graph' in Question 3 and highlight key modifications we'll have to make to the web-graph when performing PageRank. To start, we'll look at what goes wrong when we try to repeat our strategy from question 3 on a 'not nice' graph.
# MAGIC 
# MAGIC __`Additional References:`__ http://pi.math.cornell.edu/~mec/Winter2009/RalucaRemus/Lecture3/lecture3.html
# MAGIC 
# MAGIC ### Q4 Tasks:
# MAGIC 
# MAGIC * __a) code + short response:__ Run the provided code to create and plot our 'not nice' graph. Fill in the missing code to compute its transition matrix & run the power iteration method from question 3. What is wrong with what you see? [__`HINT:`__ _there is a visible underlying reason that it isn't converging... try adding up the probabilities in the state vector after each iteration._]
# MAGIC 
# MAGIC * __b) short response:__  Identify the dangling node in this 'not nice' graph and explain how this node causes the problem you described in 'a'. How could we modify the transition matrix after each iteration to prevent this problem?
# MAGIC 
# MAGIC * __c) short response:__ What does it mean for a graph to be irreducible? Is the webgraph naturally irreducible? Explain your reasoning briefly.
# MAGIC 
# MAGIC * __d) short response:__ What does it mean for a graph to be aperiodic? Is the webgraph naturally aperiodic? Explain your reasoning briefly.
# MAGIC 
# MAGIC * __e) short response:__ What modification to the webgraph does PageRank make in order to guarantee aperiodicity and irreducibility? Interpret this modification in terms of our random surfer analogy.

# COMMAND ----------

# MAGIC %md ### Q4 Student Answers:
# MAGIC > __a)__ **Run the provided code to create and plot our 'not nice' graph. Fill in the missing code to compute its transition matrix & run the power iteration method from question 3. What is wrong with what you see? [__`HINT:`__ _there is a visible underlying reason that it isn't converging... try adding up the probabilities in the state vector after each iteration._]**  
# MAGIC >  
# MAGIC > We're losing probability mass with each iteration because Node E is dangling. In the transition matrix, it has a 0 probability of going to each node on the next hop. So probability mass that goes INTO node E on each turn is lost and our overall probability shrinks with each turn. 
# MAGIC 
# MAGIC > __b)__ **Identify the dangling node in this 'not nice' graph and explain how this node causes the problem you described in 'a'. How could we modify the transition matrix after each iteration to prevent this problem?**  
# MAGIC >  
# MAGIC > The dangling node is node E. As mentioned in my answer to 4a, we're losing probability mass that comes into E because E has no outgoing connections. We could address this loss of mass by dividing each row of the transition by the sum of the weights in the state vector, forcing the probabilities for each node back to 1.
# MAGIC 
# MAGIC > __c)__ **What does it mean for a graph to be irreducible? Is the webgraph naturally irreducible? Explain your reasoning briefly.**  
# MAGIC >  
# MAGIC >  Irreducible just means there's a path from every node to every other node. Is the webgraph naturally irreducible? If by this you mean the graph of pages on the internet is naturally irreducible, then I think the answer is "no". It's possible to visit web pages with no outgoing links, creating a dangling node, resulting in an graph that is not irreducible. 
# MAGIC 
# MAGIC 
# MAGIC > __d)__ **What does it mean for a graph to be aperiodic? Is the webgraph naturally aperiodic? Explain your reasoning briefly.**   
# MAGIC >  
# MAGIC > It means that there's no common denominator of all the cycle lengths in that graph greater than 1. For example, if every cycle in a particular graph was 2, 4, 6 and 8, then that graph would be periodic because they are multiples of 2. 2 is greater than 1, so the graph would be periodic. Is the web graph naturally aperiodic. I would think so. There are so many possible cycle lengths on the web, it's hard to believe they have a common denominator greater than 1. 
# MAGIC 
# MAGIC > __e)__ **What modification to the webgraph does PageRank make in order to guarantee aperiodicity and irreducibility? Interpret this modification in terms of our random surfer analogy.**  
# MAGIC >  
# MAGIC >  Two modifications. If a web page is a dangling, rather than a 0 probability of moving to another node, those probabilities are replaced with an equal probability of moving to all nodes. This is equivalent to a web surfer choosing a random destination from all the web pages on the internet as their next destination. This is teleportation. The second modification also uses teleportation. x% of the time, for non-dangling nodes, they will be transported to a random node with equal probability. The other (1-x%), they will visit one of the outlinks of the node they are on with equal probability. The two conditions together guarantee aperiodicity and irreducibility. 

# COMMAND ----------

# part a - run this code to create a second toy graph (RUN THIS CELL AS IS)
TOY2_GRAPH = {'nodes':['A', 'B', 'C', 'D', 'E'],
              'edges':[('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'D'), 
                       ('B', 'E'), ('C', 'A'), ('C', 'E'), ('D', 'B')]}

# COMMAND ----------

# part a - simple visualization of our test graph using nx (RUN THIS CELL AS IS)
G = nx.DiGraph()
G.add_nodes_from(TOY2_GRAPH['nodes'])
G.add_edges_from(TOY2_GRAPH['edges'])
display(nx.draw(G, pos=nx.circular_layout(G), with_labels=True, alpha = 0.5))

# COMMAND ----------

# part a - run 10 steps of the power iteration method here
# HINT: feel free to use the functions get_adj_matr() and power_iteration() you wrote above
################ YOUR CODE HERE #################

TOY_ADJ_MATR_2 = get_adj_matr(TOY2_GRAPH)
transition_matrix_2 = TOY_ADJ_MATR_2.to_numpy()
transition_matrix_2 = transition_matrix_2/transition_matrix_2.sum(axis = 1)[:, None]
transition_matrix_2 = np.nan_to_num(transition_matrix_2)

# provide equal weights to start. Seems fairer!
n_nodes = len(TOY2_GRAPH['nodes'])
xInit_2 = np.full((n_nodes,),1/n_nodes)

states_2 =  power_iteration(xInit_2, transition_matrix_2, 10, verbose = True)
# print(transition_matrix_2)
# print(TOY_ADJ_MATR_2)

################ (END) YOUR CODE #################

# COMMAND ----------

transition_matrix_2

# COMMAND ----------



# COMMAND ----------

# MAGIC %md # About the Data
# MAGIC The main dataset for this data consists of a subset of a 500GB dataset released by AWS in 2009. The data includes the source and metadata for all of the Wikimedia wikis. You can read more here: 
# MAGIC > https://aws.amazon.com/blogs/aws/new-public-data-set-wikipedia-xml-data. 
# MAGIC 
# MAGIC As in previous homeworks we'll be using a 2GB subset of this data, which is available to you in this dropbox folder: 
# MAGIC > https://www.dropbox.com/sh/2c0k5adwz36lkcw/AAAAKsjQfF9uHfv-X9mCqr9wa?dl=0. 
# MAGIC 
# MAGIC Use the cells below to download the wikipedia data and a test file for use in developing your PageRank implementation(note that we'll use the 'indexed out' version of the graph) and to take a look at the files.

# COMMAND ----------

dbutils.fs.ls(DATA_PATH)

# COMMAND ----------

# open test_graph.txt file to see format (RUN THIS CELL AS IS)
with open('/dbfs/mnt/mids-w261/HW5/test_graph.txt', "r") as f_read:
  for line in f_read:
    print(line)

# COMMAND ----------

# load the data into Spark RDDs for convenience of use later (RUN THIS CELL AS IS)
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
testRDD = sc.textFile(DATA_PATH +'test_graph.txt')
indexRDD = sc.textFile(DATA_PATH + '/indices.txt')
wikiRDD = sc.textFile(DATA_PATH + '/all-pages-indexed-out.txt')

# COMMAND ----------

# MAGIC %sh
# MAGIC cat /dbfs/mnt/mids-w261/HW5/test_graph.txt

# COMMAND ----------

# display testRDD (RUN THIS CELL AS IS)
testRDD.take(10)

# COMMAND ----------

# display indexRDD (RUN THIS CELL AS IS)
indexRDD.take(30)

# COMMAND ----------

#testRDD.map(lambda x: x.split('\t')).flatMap(lambda x: [x[0], ast.literal_eval(x[1]).keys()]).collect()  #.map(lambda x: (x[1],x[0]) )

testRDD.map(lambda x: x.split('\t')).flatMap(lambda x: [x[0]] +  list(ast.literal_eval(x[1]).keys())).distinct().count()

#night fishing is good tour, 2008 in sapporo x2
#night fishing x3
#indexRDD.map(lambda x: x.split('\t')).map(lambda x: (x[1],x[0]) ).filter(lambda x: x[0] == '2921').collect()
# this brings back: https://en.wikipedia.org/wiki/%22Fish_Alive%22_30min.,_1_Sequence_by_6_Songs_Sakanaquarium_2009_@_Sapporo

#indexRDD.map(lambda x: x.split('\t')).map(lambda x: (x[1],x[0]) ).filter(lambda x: x[0] == '11777840').collect()
# 3 outlinks, results say it should be 2, but only 2 in the body of the article: Out[22]: [('11777840', 'Shin-shiro (album)')]

#indexRDD.map(lambda x: x.split('\t')).map(lambda x: (x[1],x[0]) ).filter(lambda x: x[0] == '13636570').collect()
# 3 outlinks, but only 2 in the body of the article: [('13636570', 'Victor Entertainment')]

#indexRDD.map(lambda x: x.split('\t')).map(lambda x: (x[1],x[0]) ).filter(lambda x: 'Sapporo' in x[1] and x).collect()
# 3 outlinks, but only 2 in the body of the article: [('13636570', 'Victor Entertainment')]

# COMMAND ----------

# display wikiRDD (RUN THIS CELL AS IS)
wikiRDD.take(10)

# COMMAND ----------

# MAGIC %md # Question 5: EDA part 1 (number of nodes)
# MAGIC 
# MAGIC As usual, before we dive in to the main analysis, we'll peform some exploratory data anlysis to understand our dataset. Please use the test graph that you downloaded to test all your code before running the full dataset.
# MAGIC 
# MAGIC ### Q5 Tasks:
# MAGIC * __a) short response:__ In what format is the raw data? What does the first value represent? What does the second part of each line represent? [__`HINT:`__ _no need to go digging here, just visually inspect the outputs of the head commands that we ran after loading the data above._]
# MAGIC 
# MAGIC * __b) code + short response:__ Run the provided bash command to count the number of records in the raw dataset. Explain why this is _not_ the same as the number of total nodes in the graph.
# MAGIC 
# MAGIC * __c) code:__ In the space provided below write a Spark job to count the _total number_ of nodes in this graph. 
# MAGIC 
# MAGIC * __d) short response:__ How many dangling nodes are there in this wikipedia graph? [__`HINT:`__ _you should not need any code to answer this question._]

# COMMAND ----------

# MAGIC %md ### Q5 Student Answers:
# MAGIC > __a)__ **In what format is the raw data? What does the first value represent? What does the second part of each line represent?**  
# MAGIC >  
# MAGIC > The first part of each line is a node id. The second part of each line is a dictionary of that node's outlinks. The keys in each dictionary are the node ids of the outlinks and the values are the number of outlinks from that node to each particular outlink. `'4': 3` in `5	{'4': 3, '2': 1, '6': 1}` means there are 3 links from node 5 to node 4. 
# MAGIC 
# MAGIC > __b)__ **Run the provided bash command to count the number of records in the raw dataset. Explain why this is _not_ the same as the number of total nodes in the graph.**  
# MAGIC >  
# MAGIC > If a node doesn't have an outlink to another node, it's node id will not appear as a 'key' (if you view the left most value as a key) in the RDD. It may, however, be a dangling node that's pointed TO by one of the other nodes, and therefore in one of the RDD's dictionaries. 
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC > __d)__ **How many dangling nodes are there in this wikipedia graph?**  
# MAGIC >  
# MAGIC > The number of dangling nodes is the total number of nodes minus number of records. The reason for this is that dangling nodes have no records of their neighbors. In this case, we have 15,192,277 total nodes minus 5,781,290 records which equal 9,410,987 dangling nodes. 

# COMMAND ----------

15192277 -  5781290

# COMMAND ----------

# part b - count the number of records in the raw data (RUN THIS CELL AS IS)
# 5781290
print(wikiRDD.count())

# COMMAND ----------

# part c - write your Spark job here (compute total number of nodes)
def count_nodes(dataRDD):
    """
    Spark job to count the total number of nodes.
    Returns: integer count 
    """    
    ############## YOUR CODE HERE ###############

    totalCount = dataRDD.map(lambda x: x.split('\t')).flatMap(lambda x: [x[0]] +  list(ast.literal_eval(x[1]).keys())).distinct().count()
    
    
    ############## (END) YOUR CODE ###############   
    return totalCount

# COMMAND ----------

# part c - run your counting job on the test file (RUN THIS CELL AS IS)
start = time.time()
tot = count_nodes(testRDD)
print(f'... completed job in {time.time() - start} seconds.')
print(f'Total Nodes: {tot}')

# COMMAND ----------

# part c - run your counting job on the full file (RUN THIS CELL AS IS)
start = time.time()
tot = count_nodes(wikiRDD)
print(f'... completed job in {time.time() - start} seconds.')
print(f'Total Nodes: {tot}')

# COMMAND ----------

# MAGIC %md # Question 6 - EDA part 2 (out-degree distribution)
# MAGIC 
# MAGIC As you've seen in previous homeworks the computational complexity of an implementation depends not only on the number of records in the original dataset but also on the number of records we create and shuffle in our intermediate representation of the data. The number of intermediate records required to update PageRank is related to the number of edges in the graph. In this question you'll compute the average number of hyperlinks on each page in this data and visualize a distribution for these counts (the out-degree of the nodes). 
# MAGIC 
# MAGIC ### Q6 Tasks:
# MAGIC * __a) code:__ In the space provided below write a Spark job to stream over the data and compute all of the following information:
# MAGIC  * count the out-degree of each non-dangling node and return the names of the top 10 pages with the most hyperlinks
# MAGIC  * find the average out-degree for all non-dangling nodes in the graph
# MAGIC  * take a 1000 point sample of these out-degree counts and plot a histogram of the result. 
# MAGIC  
# MAGIC  
# MAGIC * __b) short response:__ In the context of the PageRank algorithm, how is information about a node's out degree used?
# MAGIC 
# MAGIC * __c) short response:__ What does it mean if a node's out-degree is 0? In PageRank how will we handle these nodes differently than others?
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md ### Q6 Student Answers:
# MAGIC 
# MAGIC > __b)__ **In the context of the PageRank algorithm, how is information about a node's out degree used?**   
# MAGIC >  
# MAGIC > At each iteration, each node's PageRank is divided equally by its out degree and assigned to each of its out nodes. 
# MAGIC 
# MAGIC > __c)__ **What does it mean if a node's out-degree is 0? In PageRank how will we handle these nodes differently than others?**  
# MAGIC >  
# MAGIC > It means that node is dangling and has no out nodes. PageRank assigns each dangling node a a 1/N weight, where N is the number of nodes in the graph, effectively allowing it to teleport, with equal probability, to any node in the network on the next iteration. 

# COMMAND ----------

# part a - write your Spark job here (compute average in-degree, etc)
def count_degree(dataRDD, n):
    """
    Function to analyze out-degree of nodes in a a graph.
    Returns: 
        top  - (list of 10 tuples) nodes with most edges
        avgDegree - (float) average out-degree for non-dangling nodes
        sampledCounts - (list of integers) out-degree for n randomly sampled non-dangling nodes
    """
    # helper func
    def parse(line):
        node, edges = line.split('\t')
        return (node, ast.literal_eval(edges))
    
    ############## YOUR CODE HERE ###############

    tempRDD = dataRDD.map(lambda x: parse(x)).map(lambda x: (x[0], ((np.sum(list(x[1].values()))), 1)) ).cache()
  
    seqOp = lambda x, y: (x[0] + y[1][0],  x[1] + y[1][1])
    combOp = lambda p0, p1: (p0[0] + p1[0], p0[1] + p1[1])
    
    tot_out, tot_nodes = tempRDD.aggregate( (0,0), seqOp, combOp )
    
    avgDegree = tot_out/tot_nodes
    top = tempRDD.sortBy(lambda x: x[1][0], ascending=False).map(lambda x: (x[0], x[1][0])).take(10)
    sampledCounts = tempRDD.map(lambda x: x[1][0]).takeSample(False, 1000)

    
    ############## (END) YOUR CODE ###############
    #return avgDegree, top, sampledCounts
  
    return top, avgDegree, sampledCounts

# COMMAND ----------

testRDD.collect()

# COMMAND ----------

#testRDD.map(lambda x: parse(x)).map(lambda x: (x[0], ((np.sum(list(x[1].values()))), 1)) ).collect()

# COMMAND ----------

# part a - run your job on the test file (RUN THIS CELL AS IS)
start = time.time()
test_results = count_degree(testRDD,10)
print(f"... completed job in {time.time() - start} seconds")
print("Average out-degree: ", test_results[1])
print("Top 10 nodes (by out-degree:)\n", test_results[0])

# COMMAND ----------

# part a - plot results from test file (RUN THIS CELL AS IS)
plt.hist(test_results[2], bins=10)
plt.title("Distribution of Out-Degree")
display(plt.show())

# COMMAND ----------

# part a - run your job on the full file (RUN THIS CELL AS IS)
start = time.time()
full_results = count_degree(wikiRDD,1000)

print(f"... completed job in {time.time() - start} seconds")
print("Average out-degree: ", full_results[1])
print("Top 10 nodes (by out-degree:)\n", full_results[0])

# COMMAND ----------

# part a - plot results from full file (RUN THIS CELL AS IS)
plt.hist(full_results[2], bins=50)
plt.title("Distribution of Out-Degree")
display(plt.show())

# COMMAND ----------

# MAGIC %md # Question 7 - PageRank part 1 (Initialize the Graph)
# MAGIC 
# MAGIC One of the challenges of performing distributed graph computation is that you must pass the entire graph structure through each iteration of your algorithm. As usual, we seek to design our computation so that as much work as possible can be done using the contents of a single record. In the case of PageRank, we'll need each record to include a node, its list of neighbors and its (current) rank. In this question you'll initialize the graph by creating a record for each dangling node and by setting the initial rank to 1/N for all nodes. 
# MAGIC 
# MAGIC __`NOTE:`__ Your solution should _not_ hard code \\(N\\).
# MAGIC 
# MAGIC ### Q7 Tasks:
# MAGIC * __a) short response:__ What is \\(N\\)? Use the analogy of the infinite random web-surfer to explain why we'll initialize each node's rank to \\(\frac{1}{N}\\). (i.e. what is the probabilistic interpretation of this choice?)
# MAGIC 
# MAGIC * __b) short response:__ Will it be more efficient to compute \\(N\\) before initializing records for each dangling node or after? Explain your reasoning.
# MAGIC 
# MAGIC * __c) code:__ Fill in the missing code below to create a Spark job that:
# MAGIC   * parses each input record
# MAGIC   * creates a new record for any dangling nodes and sets it list of neighbors to be an empty set
# MAGIC   * initializes a rank of 1/N for each node
# MAGIC   * returns a pair RDD with records in the format specified by the docstring
# MAGIC 
# MAGIC 
# MAGIC * __d) code:__ Run the provided code to confirm that your job in `part a` has a record for each node and that your should records match the format specified in the docstring and the count should match what you computed in question 5. [__`TIP:`__ _you might want to take a moment to write out what the expected output should be fore the test graph, this will help you know your code works as expected_]
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md ### Q7 Student Answers:
# MAGIC 
# MAGIC > __a)__ Type your answer here! 
# MAGIC 
# MAGIC > __b)__ Type your answer here! 

# COMMAND ----------

testRDD.collect()

# COMMAND ----------

# part c - job to initialize the graph (RUN THIS CELL AS IS)
def initGraph(dataRDD):
    """
    Spark job to read in the raw data and initialize an 
    adjacency list representation with a record for each
    node (including dangling nodes).
    
    Returns: 
        graphRDD -  a pair RDD of (node_id , (score, edges))
        
    NOTE: The score should be a float, but you may want to be 
    strategic about how format the edges... there are a few 
    options that can work. Make sure that whatever you choose
    is sufficient for Question 8 where you'll run PageRank.
    """
    ############## YOUR CODE HERE ###############
    
    """
    MY NOTES:
    - EASY PART: FOR EACH NON-DANGLING NODE (HAS A RECORD) IN THE MAP PHASE, EMIT:
      (NODE_ID, set of tuples of form (outlink_node, outlink_count))
    - EXAMPLE: ('A', {('B', 1), ('C', 2)})
    - HOW TO HANDLE DANGLING NODES?
    - For every outlink node, emit (node_id, {empty_set}). For example, if 'A' was a destination node this time, emit:
    - ('A', {})
    
    
    
    
    
    
    
    """
    
    
    
    # write any helper functions here
    
    def parse(line):
        node, edges = line.split('\t')
        return (node, ast.literal_eval(edges))

    def emit_outlinks(node_outlinks_tuple):

        node, outlinks = node_outlinks_tuple

        #node_outlink_list = []
        yield (node, set(list(outlinks.items())))

        for o in outlinks.keys():
            yield (o, set() )

    def emit_edges(line, init_value):
        # line is tuple, first element is reference node. second is a SET of tuples. Each of THOSE tuples in turn 
        # represent the reference node's outlinks along with the number of those outlinks from that node. 
        ref_node, outlink_set = line
        
        edges = [] # list of unique edges
        #edge_count = 0 # count of edges, including repeated ones
        # edge_count will make distribution of page rank easier in Q8
        
        for o in outlink_set:
            edges.append( ((ref_node, o[0]), o[1]) )
            #edge_count += int(o[i])
        return (ref_node, (init_value,  edges))
    
    
   
    
    # write your main Spark code here
    
    
   
    
    ############## YOUR CODE HERE ###############

    tempRDD = dataRDD.map(lambda x: parse(x)).flatMap(emit_outlinks).reduceByKey(lambda x, y: x.union(y)).cache()
    N = tempRDD.count()
    graphRDD = tempRDD.map(lambda x: emit_edges(x, 1/N)).cache()
    
    
    
    
    
    ############## (END) YOUR CODE ##############
    
    return graphRDD

# COMMAND ----------

#alist

# COMMAND ----------

def parse(line):
    node, edges = line.split('\t')
    return (node, ast.literal_eval(edges))
  
def emit_edges(node_edges_tuple):

    node, edges = node_edges_tuple

    #node_outlink_list = []
    yield (node, set(list(edges.items())))

    for e in edges.keys():
        yield (e, set() )

        
alist = testRDD.map(parse).flatMap(emit_edges).reduceByKey(lambda x, y: x.union(y)).collect()
#testRDD.map(parse).flatMap(emit_edges).reduceByKey(lambda x, y: x.union(y)).collect()

# COMMAND ----------

adict = {'4': 3, '2': 1, '6': 1}
for e in adict.keys():
  print(e, type({}))

# COMMAND ----------

alist

# COMMAND ----------

type(alist[1][1])

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# part c - run your Spark job on the test graph (RUN THIS CELL AS IS)
start = time.time()
testGraph = initGraph(testRDD).collect()
print(f'... test graph initialized in {time.time() - start} seconds.')
testGraph

# COMMAND ----------

# part c - run your code on the main graph (RUN THIS CELL AS IS)
start = time.time()
wikiGraphRDD = initGraph(wikiRDD)
print(f'... full graph initialized in {time.time() - start} seconds')

# COMMAND ----------

# part c - confirm record format and count (RUN THIS CELL AS IS)
start = time.time()
print(f'Total number of records: {wikiGraphRDD.count()}')
print(f'First record: {wikiGraphRDD.take(1)}')
print(f'... initialization continued: {time.time() - start} seconds')

# COMMAND ----------

#wikiGraphRDD.take(5)

# COMMAND ----------

# MAGIC %md # Question 8 - PageRank part 2 (Iterate until convergence)
# MAGIC 
# MAGIC Finally we're ready to compute the page rank. In this last question you'll write a Spark job that iterates over the initialized graph updating each nodes score until it reaches a convergence threshold. The diagram below gives a visual overview of the process using a 5 node toy graph. Pay particular attention to what happens to the dangling mass at each iteration.
# MAGIC 
# MAGIC <img src='https://github.com/kyleiwaniec/w261_assets/blob/master/images/HW5/PR-illustrated.png?raw=true' width=50%>
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC __`A Note about Notation:`__ The formula above describes how to compute the updated page rank for a node in the graph. The $P$ on the left hand side of the equation is the new score, and the $P$ on the right hand side of the equation represents the accumulated mass that was re-distributed from all of that node's in-links. Finally, $|G|$ is the number of nodes in the graph (which we've elsewhere refered to as $N$).
# MAGIC 
# MAGIC ### Q8 Tasks:
# MAGIC * __a) short response:__ In terms of the infinite random walk analogy, interpret the meaning of the first term in the PageRank calculation: $\alpha * \frac{1}{|G|}$
# MAGIC 
# MAGIC * __b) short response:__ In the equation for the PageRank calculation above what does $m$ represent and why do we divide it by $|G|$?
# MAGIC 
# MAGIC * __c) short response:__ Keeping track of the total probability mass after each update is a good way to confirm that your algorithm is on track. How much should the total mass be after each iteration?
# MAGIC 
# MAGIC * __d) code:__ Fill in the missing code below to create a Spark job that take the initialized graph as its input then iterates over the graph and for each pass:
# MAGIC   * reads in each record and redistributes the node's current score to each of its neighbors
# MAGIC   * uses an accumulator to add up the dangling node mass and redistribute it among all the nodes. (_Don't forget to reset this accumulator after each iteration!_)
# MAGIC   * uses an accumulator to keep track of the total mass being redistributed.( _This is just for your own check, its not part of the PageRank calculation. Don't forget to reset this accumulator after each iteration._)
# MAGIC   * aggregates these partial scores for each node
# MAGIC   * applies telportation and damping factors as described in the formula above.
# MAGIC   * combine all of the above to compute the PageRank as described by the formula above.
# MAGIC   * 
# MAGIC   
# MAGIC    __WARNING:__ Some pages contain multiple hyperlinks to the same destination, please take this into account when redistributing the mass.
# MAGIC 
# MAGIC  
# MAGIC __`NOTE:`__ Please observe scalability best practices in the design of your code & comment your work clearly. You will be graded on both the clarity and the design.

# COMMAND ----------

# MAGIC %md ### Q8 Student Answers:
# MAGIC 
# MAGIC > __a)__ Type your answer here!
# MAGIC 
# MAGIC > __b)__ Type your answer here! 
# MAGIC 
# MAGIC > __c)__ Type your answer here! 

# COMMAND ----------

# part d - provided FloatAccumulator class (RUN THIS CELL AS IS)

from pyspark.accumulators import AccumulatorParam

class FloatAccumulatorParam(AccumulatorParam):
    """
    Custom accumulator for use in page rank to keep track of various masses.
    
    IMPORTANT: accumulators should only be called inside actions to avoid duplication.
    We stringly recommend you use the 'foreach' action in your implementation below.
    """
    def zero(self, value):
        return value
    def addInPlace(self, val1, val2):
        return val1 + val2

# COMMAND ----------

# ('9', (0.09090909090909091, [(('9', '5'), 1), (('9', '2'), 1)])),
#  ('10', (0.09090909090909091, [(('10', '5'), 1)])),
#  ('4', (0.09090909090909091, [(('4', '2'), 1), (('4', '1'), 1)])),
#  ('1', (0.09090909090909091, [])),

# COMMAND ----------

# part d - job to run PageRank (RUN THIS CELL AS IS)
def runPageRank(graphInitRDD, alpha = 0.15, maxIter = 10, verbose = True):
    """
    Spark job to implement page rank
    Args: 
        graphInitRDD  - pair RDD of (node_id , (score, edges))
        alpha         - (float) teleportation factor
        maxIter       - (int) stopping criteria (number of iterations)
        verbose       - (bool) option to print logging info after each iteration
    Returns:
        steadyStateRDD - pair RDD of (node_id, pageRank)
    """
    # teleportation:
    a = sc.broadcast(alpha)
    
    # damping factor:
    d = sc.broadcast(1-a.value)
    
    # initialize accumulators for dangling mass & total mass
    mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    totAccum = sc.accumulator(0.0, FloatAccumulatorParam())
    
    ############## YOUR CODE HERE ###############
    
    # write your helper functions here, 
    # please document the purpose of each clearly 
    # for reference, the master solution has 5 helper functions.

    def add_edge_count(record):
        # add total # of edges, inclusive of repeats, between page_rank and edge list, so we're not recounting it on each loop
        # add a field to store record level missing mass for dangling nodes
        node_id, page_rank, edges = record[0], record[1][0], record[1][1]
        
        edge_count = 0
        for e in edges:
            edge_count += int(e[1])
            
        new_record = (node_id, (page_rank, edge_count, edges))
        return new_record

      
      
    def dist_pr(record):
        """
        - first phase map. for each of edge in record:
        - emit (outlink_node_id, outlink_node_page_rank_share) 
        - if record has no edge, add weight to missing mass accumulator
        - emit record with page_rank set to 0 so node page rank starts from 0 in the reducer
        
        incoming data (record)
        - a node, it's page rank number of edges and edge list
        - of the form ('4', (0.09090909090909091, 2, [(('4', '2'), 1), (('4', '1'), 1)])) 
        - (node_id, (page_rank, [((node_id, edge_id_1), #edge_1), ((node_id, edge_id_2), , #edge_2), etc.  ])
        
        outgoing data, 
        - each edge's share of node_id's page_rank (edge_id_1, pr_1), (edge_id_2, pr_2), etc.
        - the incoming record with its page_rank reset to 0
        
        """
        
        node_id, pr, edge_count, edges = record[0], record[1][0], record[1][1], record[1][2]
        
        for e in edges:
            # e[0][1] is the outlink node, e[1] is the number of edges TO that outlink node
            yield (e[0][1], e[1]/edge_count*pr)
            
#         # if edge_count = 0 just emit record and let accumulator function handle. Only records to leave this function
#         # within original page_rank weights not zeroed out
#         if edge_count == 0:
#             yield record
#         else:
          # emit record with page_rank reset to 0
         
        yield (node_id, (0.0, edge_count, edges))
    
    
    
    
# ('1', 0.045454545454545456),
# ('1', (0.09090909090909091, 0, []))

    
    def updateMM_and_danglers(record, mm):
        
        # check if this is page_rank or graph structure node
        if isinstance(record[1], tuple): 
            # check if it's a dangling node with 0 edges
            if record[1][1] == 0:
                # add dangling weight to missing mass accumulator
                mm.add(record[1][1])
                # yield graph node structure with page_rank set to 0
                yield ( record[0], (0.0, 0, []))
        
        # if it's a non-dangling node, just pass the record through
        else: 
            yield record
  
  
  
    def prReducer(value_0, value_1):
        """
        Incoming stream from mapper has 2 types of key-value pairs. Both have keys that are node id's. 
        Most of these key-value pairs were the edges output by the mapper on the last step along with the page_rank
        value they received from their source node. 
        
        Some of these key-value pairs are the graph structure of individual nodes. One key aspect of all of these is that 
        all of their page_rank value should have been set to 0. This reducer will add to that value based the incoming
        page_rank value on this step. 
        
        Example key-page_rank records:
        ('1', 0.045)
        ('2', 0.091)
        
        Example graph records:
        ('1', (0.0, 0, [])) -> empty edge list
        ('2', (0.0, 1, [(('2', '3'), 1)]))  -> 1 entry in edge list
        
        Reducer updates the graph with the total page_rank for each node. Challenge is to identify the node type and account for it properly. 
        """
        
        
        
        # if value_0 is a tuple, it's a graph record. value_1 must be a page_rank value
        # add value_1's page_rank to the graph record total. a `yield` statement won't work here. Needs to be return. 
        
        if isinstance(value_0, tuple):
         
            return ( value_0[0] + value_1, value_0[1], value_0[2])
        
        # same as first condition, but with value_1 as graph record and value_0 as page_rank value
        elif isinstance(value_1, tuple):
          
            return ( value_1[0] + value_0, value_1[1], value_1[2])
        
        # if both are page_rank values, add them
        else:
          
            return value_0 + value_1
    
    
    def redistribute_teleport(record, N, mm):

        p = record[1][0]
        p_prime = a.value/N + d.value*(mm/N + p)
        updated_record = (record[0], (p_prime, record[1][1], record[1][2]))
        return updated_record

      
     
    def get_missing_mass(record, mmAccum):
        # if an empty edge list == dangling node, add page rank of node/recod to missing mass
        if record[1][1] == 0:
            mmAccum.add(record[1][0])
    
    
    
       
    #USE totAccum on each iteration to make sure total weights at end of each pass sum to 1.0
    
    
    
    # write your main Spark Job here (including the for loop to iterate)
    # for reference, the master solution is 21 lines including comments & whitespace

    """
    - For each node_id, emit 
    
    """
    
    # add edge_counts so not calculating on each iteration
    graphInitRDD = graphInitRDD.map(add_edge_count).cache() 
    
    # calculate node counts once
    N = graphInitRDD.count()
    

    
    for i in range(maxIter):    
    
        # gather missing mass
        graphInitRDD.foreach(lambda x: get_missing_mass(x, mmAccum))
        mm = mmAccum.value
    
    
  
        graphInitRDD =  graphInitRDD.flatMap(lambda x: dist_pr(x)) \
           .reduceByKey(lambda x, y: prReducer(x, y)) \
           .map(lambda x: redistribute_teleport(x, N, mm)).cache()
    
      # set accumulator back to 0.0 
        mmAccum = sc.accumulator(0.0, FloatAccumulatorParam())

      
      
    steadyStateRDD =  graphInitRDD.map(lambda x: (x[0], x[1][0]))
      
      
      
      
      
      
      
#     steadyStateRDD =  graphInitRDD.flatMap(lambda x: dist_pr(x, mmAccum)) \
#         .foreach(lambda x: updateMM_and_danglers(x, mmAccum)).cache()
    
    
    
    
    
    ############## (END) YOUR CODE ###############
    return steadyStateRDD
    #return steadyStateRDD

# COMMAND ----------

  

# COMMAND ----------

testGraphRDD.collect()

# COMMAND ----------

test_results.collect()

# COMMAND ----------

nIter = 20
testGraphRDD = initGraph(testRDD)
start = time.time()
test_results= runPageRank(testGraphRDD, alpha = 0.15, maxIter = nIter, verbose = False)

# COMMAND ----------

test_results.collect()

# COMMAND ----------

tot = 0
for atup in test_results.collect():
    tot += atup[1][0]
print(tot)

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# part d - run PageRank on the test graph (RUN THIS CELL AS IS)
# NOTE: while developing your code you may want turn on the verbose option
nIter = 20
testGraphRDD = initGraph(testRDD)
start = time.time()
test_results = runPageRank(testGraphRDD, alpha = 0.15, maxIter = nIter, verbose = False)
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
test_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

# MAGIC %md __`expected results for the test graph:`__
# MAGIC ```
# MAGIC [(2, 0.3620640495978871),
# MAGIC  (3, 0.333992700474142),
# MAGIC  (5, 0.08506399429624555),
# MAGIC  (4, 0.06030963508473455),
# MAGIC  (1, 0.04255740809817991),
# MAGIC  (6, 0.03138662354831139),
# MAGIC  (8, 0.01692511778009981),
# MAGIC  (10, 0.01692511778009981),
# MAGIC  (7, 0.01692511778009981),
# MAGIC  (9, 0.01692511778009981),
# MAGIC  (11, 0.01692511778009981)]
# MAGIC ```

# COMMAND ----------

# part d - run PageRank on the full graph (RUN THIS CELL AS IS)
# NOTE: wikiGraphRDD should have been computed & cached above!
nIter = 10
start = time.time()
full_results = runPageRank(wikiGraphRDD, alpha = 0.15, maxIter = nIter, verbose = True)
print(f'...trained {nIter} iterations in {time.time() - start} seconds.')
print(f'Top 20 ranked nodes:')
full_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

# MAGIC %md __`expected results for the full data`__
# MAGIC 
# MAGIC ```
# MAGIC 
# MAGIC top_20 = [(13455888, 0.0015447247129832947),
# MAGIC  (4695850, 0.0006710240718906518),
# MAGIC  (5051368, 0.0005983856809747697),
# MAGIC  (1184351, 0.0005982073536467391),
# MAGIC  (2437837, 0.0004624928928940748),
# MAGIC  (6076759, 0.00045509400641448284),
# MAGIC  (4196067, 0.0004423778888372447),
# MAGIC  (13425865, 0.00044155351714348035),
# MAGIC  (6172466, 0.0004224002001845032),
# MAGIC  (1384888, 0.0004012895604073632),
# MAGIC  (6113490, 0.00039578924771805474),
# MAGIC  (14112583, 0.0003943847283754762),
# MAGIC  (7902219, 0.000370098784735699),
# MAGIC  (10390714, 0.0003650264964328283),
# MAGIC  (12836211, 0.0003619948863114985),
# MAGIC  (6237129, 0.0003519555847625285),
# MAGIC  (6416278, 0.00034866235645266493),
# MAGIC  (13432150, 0.00033936510637418247),
# MAGIC  (1516699, 0.00033297500286244265),
# MAGIC  (7990491, 0.00030760906265869104)]
# MAGIC  ```

# COMMAND ----------

top_20 = full_results.takeOrdered(20, key=lambda x: - x[1])

# COMMAND ----------

spark.conf.set(
  f"fs.azure.sas.{blob_container}.{storage_account}.blob.core.windows.net",
  dbutils.secrets.get(scope = secret_scope, key = secret_key)
)



# COMMAND ----------

from pyspark.sql.functions import col, max

blob_container = "w261-team-9" # The name of your container created in https://portal.azure.com
storage_account = "w261team9" # The name of your Storage account created in https://portal.azure.com
secret_scope = "w261team9" # The name of the scope created in your local computer using the Databricks CLI
secret_key = "w261team9key" # The name of the secret key created in your local computer using the Databricks CLI 
blob_url = f"wasbs://{blob_container}@{storage_account}.blob.core.windows.net"
mount_path = "/mnt/mids-w261"

# COMMAND ----------

# Save the top_20 results to disc for use later. So you don't have to rerun everything if you restart the cluster.
#tempResults_rdd = sc.parallelize(top_20)

tempResults_DF = spark.createDataFrame(data=top_20)
blob_url = 'wasbs://w261-team-9@w261team9.blob.core.windows.net'

tempResults_DF.write.mode('overwrite').parquet(f"{blob_url}/Q8_top20_JM")

# COMMAND ----------

# view record from indexRDD (RUN THIS CELL AS IS)
# title\t indx\t inDeg\t outDeg
indexRDD.take(1)

# COMMAND ----------

# map indexRDD to new format (index, name) (RUN THIS CELL AS IS)
namesKV_RDD = indexRDD.map(lambda x: (int(x.split('\t')[1]), x.split('\t')[0]))

# COMMAND ----------

# see new format (RUN THIS CELL AS IS)
namesKV_RDD.take(2)

# COMMAND ----------

# MAGIC %md # OPTIONAL
# MAGIC ### The rest of this notebook is optional and doesn't count toward your grade.
# MAGIC The indexRDD we created earlier from the indices.txt file contains the titles of the pages and thier IDs.
# MAGIC 
# MAGIC * __a) code:__ Join this dataset with your top 20 results.
# MAGIC * __b) code:__ Print the results

# COMMAND ----------

# MAGIC %md ## Join with indexRDD and print pretty

# COMMAND ----------

# part a
joinedWithNames = None
############## YOUR CODE HERE ###############

############## END YOUR CODE ###############

# COMMAND ----------

# part b
# Feel free to modify this cell to suit your implementation, but please keep the formatting and sort order.
print("{:10s}\t| {:10s}\t| {}".format("PageRank","Page id","Title"))
print("="*100)
for r in joinedWithNames:
    print ("{:6f}\t| {:10d}\t| {}".format(r[1][1],r[0],r[1][0]))

# COMMAND ----------



# COMMAND ----------

# MAGIC %md ## OPTIONAL - GraphFrames
# MAGIC GraphFrames is a graph library which is built on top of the Spark DataFrames API.
# MAGIC 
# MAGIC * __a) code:__ Using the same dataset, run the graphframes implementation of pagerank.
# MAGIC * __b) code:__ Join the top 20 results with indices.txt and display in the same format as above.
# MAGIC * __c) short answer:__ Compare your results with the results from graphframes.
# MAGIC 
# MAGIC __NOTE:__ Feel free to create as many code cells as you need. Code should be clear and concise - do not include your scratch work. Comment your code if it's not self annotating.

# COMMAND ----------

# imports
import re
import ast
import time
import numpy as np
import pandas as pd
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from graphframes import *
from pyspark.sql import functions as F

# COMMAND ----------

# load the data into Spark RDDs for convenience of use later (RUN THIS CELL AS IS)
DATA_PATH = 'dbfs:/mnt/mids-w261/HW5/'
testRDD = sc.textFile(DATA_PATH +'test_graph.txt')
indexRDD = sc.textFile(DATA_PATH + '/indices.txt')
wikiRDD = sc.textFile(DATA_PATH + '/all-pages-indexed-out.txt')

# COMMAND ----------

# MAGIC %md
# MAGIC ### You will need to generate vertices (v) and edges (e) to feed into the graph below. 
# MAGIC Use as many cells as you need for this task.

# COMMAND ----------

# Create a GraphFrame
from graphframes import *
g = GraphFrame(v, e)


# COMMAND ----------

# Run PageRank algorithm, and show results.
results = g.pageRank(resetProbability=0.15, maxIter=10)

# COMMAND ----------

start = time.time()
top_20 = results.vertices.orderBy(F.desc("pagerank")).limit(20)
print(f'... completed job in {time.time() - start} seconds.')

# COMMAND ----------

# MAGIC %%time
# MAGIC top_20.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Run the cells below to join the results of the graphframes pagerank algorithm with the names of the nodes.

# COMMAND ----------

namesKV_RDD = indexRDD.map(lambda x: (int(x.split('\t')[1]), x.split('\t')[0]))

# COMMAND ----------

namesKV_DF = namesKV_RDD.toDF()

# COMMAND ----------

namesKV_DF = namesKV_DF.withColumnRenamed('_1','id')
namesKV_DF = namesKV_DF.withColumnRenamed('_2','title')
namesKV_DF.take(1)

# COMMAND ----------

resultsWithNames = namesKV_DF.join(top_20, namesKV_DF.id==top_20.id).orderBy(F.desc("pagerank")).collect()

# COMMAND ----------

# TODO: use f' for string formatting
print("{:10s}\t| {:10s}\t| {}".format("PageRank","Page id","Title"))
print("="*100)
for r in resultsWithNames:
    print ("{:6f}\t| {:10s}\t| {}".format(r[3],r[2],r[1]))

# COMMAND ----------

# MAGIC %md ### Congratulations, you have completed HW5! Please refer to the readme for submission instructions.
# MAGIC 
# MAGIC If you would like to provide feedback regarding this homework, please use the survey at: https://docs.google.com/forms/d/e/1FAIpQLSce9feiQeSkdP43A0ZYui1tMGIBfLfzb0rmgToQeZD9bXXX8Q/viewform

# COMMAND ----------



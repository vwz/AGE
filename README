This program (AGE) implements an active learning for graph embedding framework, as proposed in the following paper.
If you use it for scientific experiments, please cite this paper:
@article{DBLP:journals/corr/CaiZC17,
  author    = {HongYun Cai and
               Vincent Wenchen Zheng and
               Kevin Chen{-}Chuan Chang},
  title     = {Active Learning for Graph Embedding},
  journal   = {CoRR},
  volume    = {abs/1705.05085},
  year      = {2017},
  url       = {https://arxiv.org/abs/1705.05085},
  timestamp = {Mon, 15 May 2017 06:49:04 GMT}
}

The code has been tested under Ubuntu 16.04 LTS with Intel Xeon(R) CPU E5-1620 @3.50GHz*8 and 16G memory.


============== *** Installation *** ============== 
python setup.py install

============== *** Requirements *** ============== 
tensorflow (>0.12)
networkx
Graph convolutional network (Kipf and Welling, ICLR 2017): https://github.com/tkipf/gcn

============== *** Data *** ==============
In order to use your own data, you have to provide

an N by N adjacency matrix (N is the number of nodes),
an N by D feature matrix (D is the number of features per node), and
an N by E binary label matrix (E is the number of classes).
Have a look at the load_data() function in utils.py for an example.

In this example, we load citation network data (Cora, Citeseer or Pubmed). The original datasets can be found here: http://linqs.cs.umd.edu/projects/projects/lbc/. In our version (see data folder) we use dataset splits provided by https://github.com/kimiyoung/planetoid (Zhilin Yang, William W. Cohen, Ruslan Salakhutdinov, Revisiting Semi-Supervised Learning with Graph Embeddings, ICML 2016) to load the whole dataset, and use the same test data as theirs.

The validation node instances are randomly sampled from the non-test nodes set. We randomly generate 10 validation sets for each dataset and the node indexes are stored in "source/datasetname/val_idxa.txt" (where a is the validation set id, range within [0,10]).

The initially labeled nodes are randomly sampled from the non-test and non-train nodes set. Given the C (the number of classes in this dataset) and a predefined L, AGE will randomly sample L nodes from each class as the initially labeled nodes (so there are C*L initial labeled nodes in total). 

============== *** Run the Program *** ==============
1. First generate the graph centrality score for each node as follows.
Command: 
python get_graph_centrality.py datasetname 
e.g., python get_graph_centrality.py citeseer
Parameteres:
datasetname: denote the dataset to process
Output:
The centality scores for each node (same order as in graph) are stored in "res/datasetname/graphcentrality/normcen"
Note:
We adopt PageRank Centrality in this work. You can try other centrality measurements by modifing function "centralissimo()" in file "get_graph_centrality.py".
2. Run the AGE algorithm to actively select nodes to label during the graph embedding process and record the MacroF1 and MicroF1 for node classification
Command:
python train_entropy_density_graphcentral_ts.py validation_id nb_initial_labelled_nodes_per_class class_nb datasetname
e.g., python train_entropy_density_graphcentral_ts.py 0 4 6 citeseer
Parameters:
validation_id: the validation set id, refering to the id listed in "source/datasetname/val_idxa.txt"
nb_initial_labelled_nodes_per_class: number of the initial labelled nodes per class, we use four in this work
class_nb: number of class for each dataset
datasetname: the name of the dataset to process


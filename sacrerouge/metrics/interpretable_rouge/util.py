import networkx as nx
import numpy
from networkx.algorithms import bipartite
from scipy import sparse
from typing import Dict, List, Tuple


def _create_graph(matches: List[Tuple[int, int, float]]) -> nx.Graph:
    G = nx.Graph()

    # Add the nodes
    source_nodes = []
    target_notes = []
    source_index_to_node = {}
    target_index_to_node = {}
    source_node_to_index = {}
    target_node_to_index = {}
    count = 1
    for i, j, _ in matches:
        if i not in source_index_to_node:
            source_nodes.append(count)
            source_index_to_node[i] = count
            source_node_to_index[count] = i
            count += 1
        if j not in target_index_to_node:
            target_notes.append(count)
            target_index_to_node[j] = count
            target_node_to_index[count] = j
            count += 1

    G.add_nodes_from(source_nodes, bipartite=0)
    G.add_nodes_from(target_notes, bipartite=1)

    # Add the edges
    for i, j, weight in matches:
        source = source_index_to_node[i]
        target = target_index_to_node[j]
        G.add_edge(source, target, weight=weight)

    return G, source_node_to_index, target_node_to_index


def _convert_matching_to_indices(matching: List[Tuple[int, int]],
                                 source_node_to_index: Dict[int, int],
                                 target_node_to_index: Dict[int, int]) -> List[Tuple[int, int]]:
    remapped_matching = []
    for i, j in matching:
        if i in source_node_to_index:
            assert i not in target_node_to_index
            assert j not in source_node_to_index
            remapped_matching.append((source_node_to_index[i], target_node_to_index[j]))
        else:
            assert j not in target_node_to_index
            remapped_matching.append((source_node_to_index[j], target_node_to_index[i]))

    return remapped_matching


def calculate_maximum_matching(matches: List[Tuple[int, int, float]], return_matching=False) -> float:
    if len(matches) == 0:
        if return_matching:
            return 0.0, []
        return 0.0

    G, source_node_to_index, target_node_to_index = _create_graph(matches)

    # Compute the matching and convert the node ids back to the indices
    matching = nx.algorithms.matching.max_weight_matching(G)
    weight = 0
    for i, j in matching:
        weight += G[i][j]['weight']
    if return_matching:
        matching = _convert_matching_to_indices(matching, source_node_to_index, target_node_to_index)
        return weight, matching
    return weight


def calculate_all_maximum_matchings(matches: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
    if len(matches) == 0:
        return []
    G, source_node_to_index, target_node_to_index = _create_graph(matches)
    matchings = enumMaximumMatching2(G)
    return [_convert_matching_to_indices(matching, source_node_to_index, target_node_to_index) for matching in matchings]


def enumMaximumMatching2(g):
    '''Similar to enumMaximumMatching() but implemented using adjacency matrix
    of graph. Slight speed boost.
    '''

    s1=set(n for n,d in g.nodes(data=True) if d['bipartite']==0)
    s2=set(g)-s1
    n1=len(s1)
    nodes=list(s1)+list(s2)

    adj=nx.adjacency_matrix(g,nodes).tolil()
    all_matches=[]

    #----------------Find one matching----------------
    # match=bipartite.hopcroft_karp_matching(g)

    # matchadj=numpy.zeros(adj.shape).astype('int')
    # for kk,vv in match.items():
    #     matchadj[nodes.index(kk),nodes.index(vv)]=1
    # matchadj=sparse.lil_matrix(matchadj)
    #
    # all_matches.append(matchadj)

    match = nx.algorithms.matching.max_weight_matching(g)
    matchadj=numpy.zeros(adj.shape).astype('int')
    for kk,vv in match:
        matchadj[nodes.index(kk),nodes.index(vv)]=1
        matchadj[nodes.index(vv),nodes.index(kk)]=1
    matchadj=sparse.lil_matrix(matchadj)

    all_matches.append(matchadj)

    #-----------------Enter recursion-----------------
    all_matches=enumMaximumMatchingIter2(adj,matchadj,all_matches,n1,None,True)

    #---------------Re-orient match arcs---------------
    all_matches2=[]
    for ii in all_matches:
        match_list=sparse.find(ii[:n1]==1)
        m1=[nodes[jj] for jj in match_list[0]]
        m2=[nodes[jj] for jj in match_list[1]]
        match_list=list(zip(m1,m2))

        all_matches2.append(match_list)
    return all_matches2



def enumMaximumMatchingIter2(adj,matchadj,all_matches,n1,add_e=None,check_cycle=True):
    '''Similar to enumMaximumMatching() but implemented using adjacency matrix
    of graph. Slight speed boost.
    '''

    #-------------------Find cycles-------------------
    if check_cycle:
        d=matchadj.multiply(adj)
        d[n1:,:]=adj[n1:,:]-matchadj[n1:,:].multiply(adj[n1:,:])

        dg=nx.from_numpy_matrix(d.toarray(),create_using=nx.DiGraph())
        cycles=list(nx.simple_cycles(dg))
        if len(cycles)==0:
            check_cycle=False
        else:
            check_cycle=True

    #if len(cycles)>0:
    if check_cycle:
        cycle=cycles[0]
        cycle.append(cycle[0])
        cycle=zip(cycle[:-1],cycle[1:])

        #--------------Create a new matching--------------
        new_match=matchadj.copy()
        for ee in cycle:
            if matchadj[ee[0],ee[1]]==1:
                new_match[ee[0],ee[1]]=0
                new_match[ee[1],ee[0]]=0
                e=ee
            else:
                new_match[ee[0],ee[1]]=1
                new_match[ee[1],ee[0]]=1

        if add_e is not None:
            for ii in add_e:
                new_match[ii[0],ii[1]]=1

        all_matches.append(new_match)

        #-----------------Form subproblems-----------------
        g_plus=adj.copy()
        g_minus=adj.copy()
        g_plus[e[0],:]=0
        g_plus[:,e[1]]=0
        g_plus[:,e[0]]=0
        g_plus[e[1],:]=0
        g_minus[e[0],e[1]]=0
        g_minus[e[1],e[0]]=0


        add_e_new=[e,]
        if add_e is not None:
            add_e_new.extend(add_e)

        all_matches=enumMaximumMatchingIter2(g_minus,new_match,all_matches,n1,add_e,check_cycle)
        all_matches=enumMaximumMatchingIter2(g_plus,matchadj,all_matches,n1,add_e_new,check_cycle)

    else:
        #---------------Find uncovered nodes---------------
        uncovered=numpy.where(numpy.sum(matchadj,axis=1)==0)[0]

        if len(uncovered)==0:
            return all_matches

        #---------------Find feasible paths---------------
        paths=[]
        for ii in uncovered:
            aa=adj[ii,:].dot(matchadj)
            if aa.sum()==0:
                continue
            paths.append((ii,int(sparse.find(aa==1)[1][0])))
            if len(paths)>0:
                break

        if len(paths)==0:
            return all_matches

        #----------------------Find e----------------------
        feas1,feas2=paths[0]
        e=(feas1,int(sparse.find(matchadj[:,feas2]==1)[0]))

        #----------------Create a new match----------------
        new_match=matchadj.copy()
        new_match[feas2,:]=0
        new_match[:,feas2]=0
        new_match[feas1,e[1]]=1
        new_match[e[1],feas1]=1

        if add_e is not None:
            for ii in add_e:
                new_match[ii[0],ii[1]]=1

        all_matches.append(new_match)

        #-----------------Form subproblems-----------------
        g_plus=adj.copy()
        g_minus=adj.copy()
        g_plus[e[0],:]=0
        g_plus[:,e[1]]=0
        g_plus[:,e[0]]=0
        g_plus[e[1],:]=0
        g_minus[e[0],e[1]]=0
        g_minus[e[1],e[0]]=0

        add_e_new=[e,]
        if add_e is not None:
            add_e_new.extend(add_e)

        all_matches=enumMaximumMatchingIter2(g_minus,matchadj,all_matches,n1,add_e,check_cycle)
        all_matches=enumMaximumMatchingIter2(g_plus,new_match,all_matches,n1,add_e_new,check_cycle)

    return all_matches

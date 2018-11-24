"""
Metrics which measure the number of edges connected to nodes.
"""
from __future__ import division, print_function
import numpy as np
from ..utils import binarize


def degrees_dir(CIJ):
    """
    Node degree is the number of links connected to the node. The indegree
    is the number of inward links and the outdegree is the number of
    outward links.
    Directed in-degree: :math:`k_i^{in} = \displaystyle\sum_{j \in N} a_{ji}`
    Directed in-degree: :math:`k_i^{out} = \displaystyle\sum_{j \in N} a_{ij}`

    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        directed binary/weighted connection matrix

    Returns
    -------
    in_degree : Nx1 :obj:`numpy.ndarray`
        node in-degree
    out_degree : Nx1 :obj:`numpy.ndarray`
        node out-degree
    deg : Nx1 :obj:`numpy.ndarray`
        node degree (in-degree + out-degree)

    Notes
    -----
    Inputs are assumed to be on the columns of the CIJ matrix.
           Weight information is discarded.
    """
    CIJ = binarize(CIJ, copy=True)  # ensure CIJ is binary
    in_degree = np.sum(CIJ, axis=0)  # indegree = column sum of CIJ
    out_degree = np.sum(CIJ, axis=1)  # outdegree = row sum of CIJ
    deg = in_degree + out_degree  # degree = indegree+outdegree
    return in_degree, out_degree, deg


def degrees_und(CIJ):
    """
    Node degree is the number of links connected to the node.
    Binary degree: :math:`k_i = \displaystyle\sum_{j \in N} a_{ijj}`
    Weighted degree: :math:`k_i^w = \displaystyle\sum_{j \in N} w_{ij}`

    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        undirected binary/weighted connection matrix

    Returns
    -------
    deg : Nx1 :obj:`numpy.ndarray`
        node degree

    Notes
    -----
    Weight information is discarded.
    """
    CIJ = binarize(CIJ, copy=True)  # ensure CIJ is binary
    return np.sum(CIJ, axis=0)


def jdegree(CIJ):
    """
    This function returns a matrix in which the value of each element (u,v)
    corresponds to the number of nodes that have u outgoing connections
    and v incoming connections.

    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        directed binary/weighted connnection matrix

    Returns
    -------
    J : ZxZ :obj:`numpy.ndarray`
        joint degree distribution matrix
        (shifted by one, replicates matlab one-based-indexing)
    J_od : int
        number of vertices with out_degree>in_degree
    J_id : int
        number of vertices with in_degree>out_degree
    J_bl : int
        number of vertices with in_degree==out_degree

    Notes
    -----
    Weights are discarded.
    """
    CIJ = binarize(CIJ, copy=True)  # ensure CIJ is binary
    n_nodes = len(CIJ)
    in_degree = np.sum(CIJ, axis=0)  # indegree = column sum of CIJ
    out_degree = np.sum(CIJ, axis=1)  # outdegree = row sum of CIJ

    # create the joint degree distribution matrix
    # note: the matrix is shifted by one, to accomodate zero in_degree and
    # out_degree in the first row/column
    # upper triangular part of the matrix has vertices with
    # out_degree>in_degree
    # lower triangular part has vertices with in_degree>out_degree
    # main diagonal has units with in_degree=out_degree

    szJ = np.max((in_degree, out_degree)) + 1
    J = np.zeros((szJ, szJ))

    for i in range(n_nodes):
        J[in_degree[i], out_degree[i]] += 1

    J_od = np.sum(np.triu(J, 1))
    J_id = np.sum(np.tril(J, -1))
    J_bl = np.sum(np.diag(J))
    return J, J_od, J_id, J_bl


def strengths_dir(CIJ):
    """
    Node strength is the sum of weights of links connected to the node. The
    instrength is the sum of inward link weights and the outstrength is the
    sum of outward link weights.

    Parameters
    ----------
    CIJ : NxN :obj:`numpy.ndarray`
        directed weighted connection matrix

    Returns
    -------
    is : Nx1 :obj:`numpy.ndarray`
        node in-strength
    os : Nx1 :obj:`numpy.ndarray`
        node out-strength
    str : Nx1 :obj:`numpy.ndarray`
        node strength (in-strength + out-strength)

    Notes
    -----
    Inputs are assumed to be on the columns of the CIJ matrix.
    """
    istr = np.sum(CIJ, axis=0)
    ostr = np.sum(CIJ, axis=1)
    return istr + ostr


def strengths(adj, flavor='auto', norm=False):
    """
    Node strength is the sum of weights of links connected to the node,
    the analog of degree in weighted networks.
    Strength: :math:`s_i = \displaystyle\sum_{j \neq i} w_{ij}`
    Normalized strength: :math:`s_i^{\prime} = \frac{1}{N - 1} \displaystyle\sum_{j \neq i} w_{ij}`

    Positive strength: :math:`s_i^+ = \displaystyle\sum_{j \neq i}^+ w_{ij}`
    Negative strength: :math:`s_i^- = - \displaystyle\sum_{j \neq i}^- w_{ij}`

    Unitary, normalized strength: :math:`s_i^{*} = s_i^{\prime +} - s_i^{\prime -} \left(\frac{s_i^{\prime -}}{s_i^{\prime +} + s_i^{\prime -}}\right)`

    Parameters
    ----------
    mat : NxN :obj:`numpy.ndarray`
        connection matrix
    flavor : {'bu', 'wu', 'bd', 'wd', 'auto'}, optional
        type of connection matrix `mat`: binary or weighted (b or w) and
        directed or undirected (d or u). 'auto' detects these from the data using
        `check_mtx_fmt`.
    norm : :obj:`bool`
        if True, normalized strength is returned. Default is False.

    Returns
    -------
    strength : Nx1 :obj:`numpy.ndarray`
        node strengths
    flav : :obj:`str`
        type of matrix fed in, if  `flavor='auto'`
    Spos : Nx1 :obj:`numpy.ndarray`
        nodal strength of positive weights, if  `flavor=''`
    Sneg : Nx1 :obj:`numpy.ndarray`
        nodal strength of positive weights
    vpos : float
        total positive weight
    vneg : float
        total negative weight
    """
    if flavor == 'auto':
        flav = check_mtx_fmt(adj)
        return flav

    adjacency = adj.copy()
    np.fill_diagonal(adjacency, 0)
    if flavor == 'wu' or (flavor == 'auto' and (flav.bin_status == 'wei' and flav.direction == 'und')):
        strength = np.sum(adjacency, axis=0)
        if norm:
            strength_norm = strength/(len(adj[0])-1)
            strength = strength_norm
        elif:
            strength = strength
        return strength
    elif flavor == 'wd' or (flavor == 'auto' and (flav.bin_status == 'wei' and flav.direction == 'dir')):
        Spos = np.sum(adjacency * (adjacency > 0), axis=0)  # positive strengths
        Sneg = np.sum(adjacency * (adjacency < 0), axis=0)  # negative strengths

        vpos = np.sum(Spos)  # total positive weight
        vneg = np.sum(Sneg)  # total negative weight
        if norm:
            #UNITARY NORMALIZED
            strength_unit_norm = (Spos / (len(adj[0])-1)) - (Sneg / (len(adj[0])-1)) (Sneg/(Sneg + Spos))
            return strength_unit_norm
        elif:
            return Spos, Sneg, vpos, vneg
    elif flavor == 'bu' or flavor == 'bd' or (flavor == 'auto' and flav.bin_status == 'bin'):
        #calculate degrees
        degree = np.sum(adj, axis=0)
        if norm:
            #NOPE
            return 'No such normalized degree exists.'
        elif:
            return degree



def strengths_und_sign(W):
    """
    Node strength is the sum of weights of links connected to the node.
    Positive strength: :math:`s_i^+ = \displaystyle\sum_{j \neq i}^+ w_{ij}`
    Negative strength: :math:`s_i^- = - \displaystyle\sum_{j \neq i}^- w_{ij}`

    Parameters
    ----------
    W : NxN :obj:`numpy.ndarray`
        undirected connection matrix with positive and negative weights

    Returns
    -------
    Spos : Nx1 :obj:`numpy.ndarray`
        nodal strength of positive weights
    Sneg : Nx1 :obj:`numpy.ndarray`
        nodal strength of positive weights
    vpos : float
        total positive weight
    vneg : float
        total negative weight
    """
    W = W.copy()
    np.fill_diagonal(W, 0)  # clear diagonal
    Spos = np.sum(W * (W > 0), axis=0)  # positive strengths
    Sneg = np.sum(W * (W < 0), axis=0)  # negative strengths

    vpos = np.sum(W[W > 0])  # positive weight
    vneg = np.sum(W[W < 0])  # negative weight
    return Spos, Sneg, vpos, vneg

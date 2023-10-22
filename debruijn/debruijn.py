#!/bin/env python3
# -*- coding: utf-8 -*-
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    A copy of the GNU General Public License is available at
#    http://www.gnu.org/licenses/gpl-3.0.html

"""Perform assembly based on debruijn graph."""

from typing import Iterator, Dict, List
import matplotlib.pyplot as plt
import itertools
import textwrap
import statistics
import argparse
import os
import sys
import networkx as nx
from pathlib import Path
from networkx import DiGraph, all_simple_paths, lowest_common_ancestor, \
    has_path, random_layout, draw, spring_layout
import matplotlib
from operator import itemgetter
import random
from random import randint
random.seed(9001)
matplotlib.use("Agg")


__author__ = "Louiza GALOU"
__copyright__ = "Universite Paris Diderot"
__credits__ = ["Louiza GALOU"]
__license__ = "GPL"
__version__ = "1.0.0"
__maintainer__ = "Louiza GALOU"
__email__ = "louizagalou59@gmail.com"
__status__ = "Developpement"


def isfile(path: str) -> Path:  # pragma: no cover
    """Check if path is an existing file.

    :param path: (str) Path to the file

    :raises ArgumentTypeError: If file does not exist

    :return: (Path) Path object of the input file
    """
    myfile = Path(path)
    if not myfile.is_file():
        if myfile.is_dir():
            msg = f"{myfile.name} is a directory."
        else:
            msg = f"{myfile.name} does not exist."
        raise argparse.ArgumentTypeError(msg)
    return myfile


def get_arguments():  # pragma: no cover
    """Retrieves the arguments of the program.

    :return: An object that contains the arguments
    """
    # Parsing arguments
    parser = argparse.ArgumentParser(description=__doc__, usage="{0} -h"
                                     .format(sys.argv[0]))
    parser.add_argument('-i', dest='fastq_file', type=isfile,
                        required=True, help="Fastq file")
    parser.add_argument('-k', dest='kmer_size', type=int,
                        default=22, help="k-mer size (default 22)")
    parser.add_argument(
        '-o',
        dest='output_file',
        type=Path,
        default=Path(
            os.curdir +
            os.sep +
            "contigs.fasta"),
        help="Output contigs in fasta file (default contigs.fasta)")
    parser.add_argument('-f', dest='graphimg_file', type=Path,
                        help="Save graph as an image (png)")
    return parser.parse_args()


def read_fastq(fastq_file: Path) -> Iterator[str]:
    """Extract reads from fastq files.

    :param fastq_file: (Path) Path to the fastq file.
    :return: A generator object that iterate the read sequences.
    """
    with open(fastq_file, 'r') as f:
        for line in f:
            yield next(f).strip()
            next(f)
            next(f)


def cut_kmer(read: str, kmer_size: int) -> Iterator[str]:
    """Cut read into kmers of size kmer_size.

    :param read: (str) Sequence of a read.
    :return: A generator object that provides the kmers (str) of size kmer_size.
    """
    for i in range(len(read) - kmer_size + 1):
        yield read[i:i + kmer_size]


def build_kmer_dict(fastq_file: Path, kmer_size: int) -> Dict[str, int]:
    """Build a dictionnary object of all kmer occurrences in the fastq file

    :param fastq_file: (str) Path to the fastq file.
    :return: A dictionnary object that identify all kmer occurrences.
    """
    kmer_dict = {}
    for sequence in read_fastq(fastq_file):
        for kmer in cut_kmer(sequence, kmer_size):
            kmer_dict[kmer] = kmer_dict.get(kmer, 0) + 1
    return kmer_dict


def build_graph(kmer_dict: Dict[str, int]) -> DiGraph:
    """Build the debruijn graph

    :param kmer_dict: A dictionnary object that identify all kmer occurrences.
    :return: A directed graph (nx) of all kmer substring and weight (occurrence).
    """
    graph = DiGraph()

    for kmer, count in kmer_dict.items():
        prefix = kmer[:-1]
        suffix = kmer[1:]

        graph.add_edge(prefix, suffix, weight=count)

    return graph


def remove_paths(graph: DiGraph, path_list: List[List[str]],
                 delete_entry_node: bool, delete_sink_node: bool) -> DiGraph:
    """Remove a list of path in a graph. A path is set of connected node in
    the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    for path in path_list:
        if not path:
            continue
        if delete_entry_node and delete_sink_node:
            graph.remove_nodes_from(path)
        elif delete_entry_node:
            graph.remove_nodes_from(path[:-1])
        elif delete_sink_node:
            graph.remove_nodes_from(path[1:])
        else:
            graph.remove_nodes_from(path[1:-1])
    return graph


def select_best_path(graph: DiGraph,
                     path_list: List[List[str]],
                     path_length: List[int],
                     weight_avg_list: List[float],
                     delete_entry_node: bool = False,
                     delete_sink_node: bool = False) -> DiGraph:
    """Select the best path between different paths

    :param graph: (nx.DiGraph) A directed graph object
    :param path_list: (list) A list of path
    :param path_length_list: (list) A list of length of each path
    :param weight_avg_list: (list) A list of average weight of each path
    :param delete_entry_node: (boolean) True->We remove the first node of a path
    :param delete_sink_node: (boolean) True->We remove the last node of a path
    :return: (nx.DiGraph) A directed graph object
    """
    # Check the stdev of weight_avg_list
    weight_stdev = statistics.stdev(weight_avg_list)
    selected_path = None
    if weight_stdev > 0:
        # Select the path with the highest weight
        max_weight_idx = weight_avg_list.index(max(weight_avg_list))
        selected_path = path_list[max_weight_idx]
    else:
        # Check the stdev of path_length
        length_stdev = statistics.stdev(path_length)
        if length_stdev > 0:
            # Select the longest path
            max_length_idx = path_length.index(max(path_length))
            selected_path = path_list[max_length_idx]
        else:
            # Randomly select a path
            random_idx = random.randint(0, len(path_list) - 1)
            selected_path = path_list[random_idx]

    # Remove all paths from the graph except the selected one
    for path in path_list:
        if path != selected_path:
            remove_path = path.copy()
            if not delete_entry_node:
                remove_path.pop(0)
            if not delete_sink_node:
                remove_path.pop(-1)
            graph.remove_nodes_from(remove_path)

    return graph


def path_average_weight(graph: DiGraph, path: List[str]) -> float:
    """Compute the weight of a path

    :param graph: (nx.DiGraph) A directed graph object
    :param path: (list) A path consist of a list of nodes
    :return: (float) The average weight of a path
    """
    return statistics.mean(
        [d["weight"] for (u, v, d) in graph.subgraph(path).edges(data=True)])


def solve_bubble(
        graph: DiGraph,
        ancestor_node: str,
        descendant_node: str) -> DiGraph:
    """Explore and solve bubble issue

    :param graph: (nx.DiGraph) A directed graph object
    :param ancestor_node: (str) An upstream node in the graph
    :param descendant_node: (str) A downstream node in the graph
    :return: (nx.DiGraph) A directed graph object
    """
    # Identify all simple paths between ancestor and descendant nodes
    paths = list(nx.all_simple_paths(graph, ancestor_node, descendant_node))

    # For each path, calculate its length and average weight
    path_lengths = [len(path) for path in paths]
    path_weights = [path_average_weight(graph, path) for path in paths]

    # Resolve bubble
    graph = select_best_path(graph, paths, path_lengths, path_weights)

    return graph


def simplify_bubbles(graph: DiGraph) -> DiGraph:
    """Detect and explode bubbles

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    bubble = False

    # Iterate over each node in the graph
    for node in graph.nodes():
        predecessors = list(graph.predecessors(node))

        # If the node has more than one predecessor, check for common ancestors
        if len(predecessors) > 1:
            for i, j in itertools.combinations(predecessors, 2):
                ancestor = nx.lowest_common_ancestor(graph, i, j)

                # If a common ancestor exists, it indicates a bubble
                if ancestor:
                    bubble = True
                    break  # Break out of the inner loop

            # Break out of the outer loop if a bubble is detected
            if bubble:
                break

    # If a bubble is detected, solve it and recursively call simplify_bubbles
    if bubble:
        graph = simplify_bubbles(solve_bubble(graph, ancestor, node))

    return graph


def solve_entry_tips(graph, starting_nodes):
    """Remove entry tips

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) List of starting nodes
    :return: (nx.DiGraph) A directed graph object without unwanted entry paths
    """


    for node in list(graph.nodes):
        all_paths_for_node = []
        predecessors = list(graph.predecessors(node))
        if len(predecessors)>1:
                for start_node in starting_nodes:
                    if node not in starting_nodes:
                        paths =  list(nx.all_simple_paths(graph, start_node, node))
                        all_paths_for_node.append(paths[0])#toujours un seul path
                if len(all_paths_for_node)>1 :
                    path_length = [len(path) for path in all_paths_for_node]
                    path_weigths = [path_average_weight(graph, all_paths_for_node[i])  if path_length[i] >1 else graph[paths[i][0]][paths[i][1]]["weight"] for i in range(len(all_paths_for_node)) ]
                    graph = select_best_path(graph, all_paths_for_node,path_length, path_weigths, delete_entry_node=True, delete_sink_node=False)
                    #graph = solve_entry_tips(graph, starting_nodes)
                    break
    return graph


def solve_out_tips(graph, ending_nodes):
    """Remove out tips

    :param graph: (nx.DiGraph) A directed graph object
    :return: (nx.DiGraph) A directed graph object
    """
    for node in list(graph.nodes):
        all_paths_for_node = []
        successors = list(graph.successors(node))
        if len(successors)>1:
                for end_node in ending_nodes:
                    if node not in ending_nodes:
                        paths =  list(nx.all_simple_paths(graph, node, end_node))
                        all_paths_for_node.append(paths[0])#toujours un seul path
                if len(all_paths_for_node)>1 :
                    path_length = [len(path) for path in all_paths_for_node]
                    path_weigths = [path_average_weight(graph, all_paths_for_node[i])  if path_length[i] >1 else graph[paths[i][0]][paths[i][1]]["weight"] for i in range(len(all_paths_for_node)) ]
                    graph = select_best_path(graph, all_paths_for_node,path_length, path_weigths, delete_entry_node=False, delete_sink_node=True)
                    #graph = solve_out_tips(graph, ending_nodes)
                    break
    return graph


def get_starting_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without predecessors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without predecessors
    """
    return [node for node, degree in graph.in_degree() if degree == 0]


def get_sink_nodes(graph: DiGraph) -> List[str]:
    """Get nodes without successors

    :param graph: (nx.DiGraph) A directed graph object
    :return: (list) A list of all nodes without successors
    """
    return [node for node, degree in graph.out_degree() if degree == 0]


def get_contigs(
        graph: DiGraph,
        starting_nodes: List[str],
        ending_nodes: List[str]) -> List:
    """Extract the contigs from the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param starting_nodes: (list) A list of nodes without predecessors
    :param ending_nodes: (list) A list of nodes without successors
    :return: (list) List of [contiguous sequence and their length]
    """
    contigs = []

    for start in starting_nodes:
        for end in ending_nodes:
            # Check for all simple paths from start to end
            for path in nx.all_simple_paths(graph, start, end):
                # Construct the contig sequence from the path
                contig = path[0]
                for i in range(1, len(path)):
                    # Add the last character of each kmer in the path
                    contig += path[i][-1]
                contigs.append((contig, len(contig)))

    return contigs


def save_contigs(contigs_list: List[str], output_file: Path) -> None:
    """Write all contigs in fasta format

    :param contig_list: (list) List of [contiguous sequence and their length]
    :param output_file: (Path) Path to the output file
    """
    with open(output_file, 'w') as f:
        for idx, (contig, length) in enumerate(contigs_list):
            header = f">contig_{idx} len={length}\n"
            f.write(header)
            f.write(textwrap.fill(contig, width=80))
            f.write("\n")


def draw_graph(graph: DiGraph, graphimg_file: Path) -> None:  # pragma: no cover
    """Draw the graph

    :param graph: (nx.DiGraph) A directed graph object
    :param graphimg_file: (Path) Path to the output file
    """
    fig, ax = plt.subplots()
    elarge = [
        (u, v) for (
            u, v, d) in graph.edges(
            data=True) if d['weight'] > 3]
    # print(elarge)
    esmall = [
        (u, v) for (
            u, v, d) in graph.edges(
            data=True) if d['weight'] <= 3]
    # print(elarge)
    # Draw the graph with networkx
    # pos=nx.spring_layout(graph)
    pos = nx.random_layout(graph)
    nx.draw_networkx_nodes(graph, pos, node_size=6)
    nx.draw_networkx_edges(graph, pos, edgelist=elarge, width=6)
    nx.draw_networkx_edges(graph, pos, edgelist=esmall, width=6, alpha=0.5,
                           edge_color='b', style='dashed')
    # nx.draw_networkx(graph, pos, node_size=10, with_labels=False)
    # save image
    plt.savefig(graphimg_file.resolve())


# ==============================================================
# Main program
# ==============================================================
def main() -> None:  # pragma: no cover
    """
    Main program function
    """
    # Get arguments
    args = get_arguments()
    # Read sequences and construct k-mer dictionary
    kmer_dict = build_kmer_dict(args.fastq_file, args.kmer_size)
    # Build the De Bruijn graph
    graph = build_graph(kmer_dict)

    # Resolve bubbles
    graph = simplify_bubbles(graph)

    # Resolve entry and exit tips
    starting_nodes = get_starting_nodes(graph)
    ending_nodes = get_sink_nodes(graph)
    graph = solve_entry_tips(graph, starting_nodes)
    graph = solve_out_tips(graph, ending_nodes)

    # Extract contigs and save them
    contigs = get_contigs(graph, starting_nodes, ending_nodes)
    save_contigs(contigs, args.output_file)

    # Optional: Plot the graph if the flag is provided
    if args.graphimg_file:
        draw_graph(graph, args.graphimg_file)


if __name__ == '__main__':  # pragma: no cover
    main()

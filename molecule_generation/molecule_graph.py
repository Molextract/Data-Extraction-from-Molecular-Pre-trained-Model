import sys, os
import random
import time
import csv
import itertools
import copy

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import pandas as pd

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import qed, MolLogP
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalogParams, FilterCatalog

from molecule_generation.utils_graph import *
from loader import mol_to_graph_data_obj_simple
from util import ExtractSubstructureContextPair
from batch import BatchSubstructContext


def get_att_points(mol):
    att_points = []
    for a in mol.GetAtoms():
        if a.GetSymbol() == '*':
            att_points.append(a.GetIdx())

    return att_points


def map_idx(idx, idx_list, mol):
    abs_id = idx_list[idx]
    neigh_idx = mol.GetAtomWithIdx(abs_id).GetNeighbors()[0].GetIdx()
    return neigh_idx


def sanitize(mol):
    try:
        smiles = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        return None
    return mol


class Molecule_Graph:
    def __init__(self, starting_smi):
        self.starting_smi = starting_smi
        self.mol = Chem.MolFromSmiles(self.starting_smi)
        self.smile_list = []
        self.smile_old_list = []

        possible_atoms = ATOM_VOCAB
        possible_motifs = FRAG_VOCAB
        possible_bonds = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                          Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
        self.atom_type_num = len(possible_atoms)
        self.motif_type_num = len(possible_motifs)
        self.possible_atom_types = np.array(possible_atoms)
        self.possible_motif_types = np.array(possible_motifs)
        self.possible_bond_types = np.array(possible_bonds, dtype=object)

        self.attach_point = Chem.MolFromSmiles('*')
        self.Na = Chem.MolFromSmiles('[Na+]')
        self.K = Chem.MolFromSmiles('[K+]')
        self.H = Chem.MolFromSmiles('[H]')

        # self.transform = ExtractSubstructureContextPair(K, l1, l2)

    def add_motif(self, ac):
        self.old_mol = copy.deepcopy(self.mol)

        cur_mol = Chem.ReplaceSubstructs(self.mol, self.attach_point, self.Na)[ac[0]]
        # print(Chem.MolToSmiles(cur_mol))
        motif = FRAG_VOCAB_MOL[ac[1]]
        att_point = FRAG_VOCAB_ATT[ac[1]]
        # print(Chem.MolToSmiles(motif))
        motif_atom = map_idx(ac[2], att_point, motif)
        motif = Chem.ReplaceSubstructs(motif, self.attach_point, self.K)[ac[2]]
        # print(Chem.MolToSmiles(motif))
        motif = Chem.DeleteSubstructs(motif, self.K)
        # print(Chem.MolToSmiles(motif))
        next_mol = Chem.ReplaceSubstructs(cur_mol, self.Na, motif, replacementConnectionPoint=motif_atom)[0]
        # print(Chem.MolToSmiles(next_mol))
        next_mol = Chem.RemoveHs(next_mol)
        # print(Chem.MolToSmiles(next_mol))
        self.mol = next_mol

    @property
    def attach_points(self, ):
        return get_att_points(self.mol)

    @property
    # delete H
    def mol_core(self):
        mol = copy.deepcopy(self.mol)
        mol = sanitize(Chem.ReplaceSubstructs(mol, att, H, replaceAll=True)[0])
        return mol

    @property
    def mol_graph(self, ):
        mol = self.mol_core
        return mol_to_graph_data_obj_simple(mol)

    def representation(self, model, device=None):
        if device is None:
            return model.forward_graph_representation(self.mol_graph)
        return model.forward_graph_representation(self.mol_graph.to(device))

    @property
    def atom_num(self):
        return self.mol_core.GetNumAtoms()

    def get_freq(self, mol_bank):
        mol_core = self.mol_core
        matched_mols = [mol for mol in mol_bank if mol.HasSubstructMatch(mol_core)]
        return len(matched_mols)

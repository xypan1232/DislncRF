# normal libaries
import os
import urllib
import gzip
import re
import collections
import sys
import zipfile
import operator
#from functools import reduce

# 3rd party (all avalible trough pip!)
import numpy as np

import matplotlib as mp
#mp.use("Agg")
from parseobo import obo_object
from matplotlib import pyplot as plt
import lmfit

################################################################################
# CONSTANTS AND PATHS
################################################################################
DEBUGGING = False
MIRBASE_VERSION = 20
# SCALING_FACTOR = 2.2275
SCALING_FACTOR = 1
# PREDICTION_FILES = ("miRanda.tsv", "TargetScan.tsv", "PicTar.tsv")

# PATHS
MASTER_DIR = "master_files"
ID_PATH = "id_dictionaries"
DATA_PATH = "data"
FIG_PATH = "figures"

# miRBase
MIR_MAPPING_ALIASES_PATH = os.path.join(ID_PATH, 'miRBase_%i_mir_aliases.tsv')
MIR_MATURE_MAPPING_ALIASES_PATH = os.path.join(ID_PATH, 'miRBase_%i_mir_aliases_only_mature.tsv')
MIRBASE_ALIASES_TXT_PATH = os.path.join(ID_PATH, "miRBase_%i_aliases.txt.gz")
MIRBASE_MIRNA_DAT_PATH = os.path.join(ID_PATH, "miRBase_%i_miRNA.dat.gz")
MIRBASE_BASE_URL = "ftp://mirbase.org/pub/mirbase/%i"

# MIR_MAPPING_UNIQUE_PATH = 'id_dictionaries/miRBase_%i_mir_unique.tsv'
# MIRBASE_MIR_ALIASES_MAPPING = 'id_dictionaries/miRBase_mir_aliases.tsv'
# STEM_LOOP_MAPPING_FILE = "id_dictionaries/miRBase_stem_loop_mapping.tsv"
# MIR_TO_STEM_LOOP_MAPPING_FILE = "id_dictionaries/miRBase_mir_to_stem_loop_mapping.tsv"


# STRING ENSP
STRING_ALIASES_100 = "http://string-db.org/newstring_download/protein.aliases.v10.txt.gz"
STRING_ALIASES_91 = "http://string91.embl.de/newstring_download/protein.aliases.v9.1.txt.gz"
STRING_ALIASES_83 = "http://string83.embl.de/newstring_download/protein.aliases.v8.3.txt.gz"

STRING_SPECIES_91 = 'http://string-db.org/newstring_download/species.v9.1.txt'
STRING_SPECIES_100 = "http://string-db.com/newstring_download/species.v10.txt"
# PubMed IDs of experiments integrated in StarBase 2.0
STARBASE_PMIDs = os.path.join(DATA_PATH, "starBase_2.0_Experiments_PubMedIDs.tsv")

#GOLDSTANDARD_FILE = os.path.join(DATA_PATH, "disease/data_disease_new.txt")
#GOLDSTANDARD_FILE = os.path.join(DATA_PATH, "disease/merge_disease.txt")
GOLDSTANDARD_FILE = os.path.join(DATA_PATH, "disease/data_disease_doid.txt")
################################################################################
# utility classes
################################################################################


class Interaction:
    """ Simple interaction class to store an interaction, i.e., a line in a master file."""

    def __init__(self, org, ent1, ent2, directed, channel, score, sources, url, comment):
        self._org = org
        self._ent1 = ent1
        self._ent2 = ent2
        self._directed = directed
        self._channel = channel
        self._score = score
        self._sources = sources
        self._url = url
        self._comment = comment

    def __str__(self):
        return '\t'.join((self._org, self._ent1, self._ent2, self._directed, self._channel, self._score, self._sources,
                          self._url, self._comment))

    def __hash__(self):
        self_ent_sorted = sorted((self._ent1, self._ent2))
        return (self._org + self_ent_sorted[0] + self_ent_sorted[1]).__hash__()

    def __eq__(self, other):
        """
        Two Interactions are equal if they are in the same organism and connect the same entities.

        :param other: The Interaction to compare to.
        :return: True if and only if the Interactions being compared are equal.
        """
        if isinstance(other, self.__class__):
            self_ent_sorted = sorted((self._ent1, self._ent2))
            other_ent_sorted = sorted((other._ent1, other._ent2))
            return (self._org == other._org) and (self_ent_sorted[0] == other_ent_sorted[0]) and \
                   (self_ent_sorted[1] == other_ent_sorted[1])
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)


class EntityType:
    """Enumerator class representing different molecular entities."""

    Protein, miRNA, ncRNA = range(3)


class InteractionType:
    """Enumerator class representing different types of interactions."""

    Protein_miRNA, Protein_ncRNA, miRNA_ncRNA, ncRNA_ncRNA = range(4)

    @staticmethod
    def entities_to_interaction_type(ent1, ent2):
        if (ent1 == EntityType.Protein and ent2 == EntityType.miRNA) or \
           (ent2 == EntityType.Protein and ent1 == EntityType.miRNA):
            return InteractionType.Protein_miRNA
        elif (ent1 == EntityType.Protein and ent2 == EntityType.ncRNA) or \
             (ent2 == EntityType.Protein and ent1 == EntityType.ncRNA):
            return InteractionType.Protein_ncRNA
        elif (ent1 == EntityType.miRNA and ent2 == EntityType.ncRNA) or \
             (ent2 == EntityType.miRNA and ent1 == EntityType.ncRNA):
            return InteractionType.miRNA_ncRNA
        elif (ent1 == EntityType.ncRNA and ent2 == EntityType.ncRNA) or \
             (ent2 == EntityType.ncRNA and ent1 == EntityType.ncRNA):
            return InteractionType.ncRNA_ncRNA
        else:
            raise Exception('Unknown interaction.')


    @staticmethod
    def interaction_type_to_string(interaction_type):
        if interaction_type == InteractionType.Protein_miRNA:
            return 'Protein-miRNA'
        elif interaction_type == InteractionType.Protein_ncRNA:
            return 'Protein-ncRNA'
        elif interaction_type == InteractionType.miRNA_ncRNA:
            return 'miRNA-ncRNA'
        elif interaction_type == InteractionType.ncRNA_ncRNA:
            return 'ncRNA-ncRNA'


################################################################################
# 'math' functions
################################################################################
def fancy_sigmoid_fun(x, a, b, k, q, m, v):
    return a + (k - a) / ((1 + q * np.exp(-b *(x-m))) ** (1 / v))


def sigmoid_fun(x, a, b, c, d):
    # Comment in the following line to get more information printed at runtime if np.exp raises a RuntimeWarning.
    #warnings.filterwarnings('error')

    # If exp raises a runtime warning due to overflow, print the warning, some additional information, raise exception
    try:
        result = (a - d) / (1 + np.exp(-1 * b * (x - c))) + d
    except RuntimeWarning as runtime_warning:
        print('x=%s, a=%s, b=%s, c=%s, d=%s' % (x, a, b, c, d))
        raise runtime_warning
    return result


def residuals(x,y, a, b, c, d):
    return y - sigmoid_fun(x, a, b, c, d)

def f_line(x, a, b):
    return a * x + b


################################################################################
# MAPPING FUNCTIONS
################################################################################
# general functions
def file_to_hash(mapping_file):
    mapping_hash = {}
    for line in open (mapping_file, 'r'):
        key, val = line.rstrip().split('\t')
        mapping_hash[key] = val
    return mapping_hash

# miRBase functions
# def infer_organism_from_mir_prefix(mir):
#     prefix_to_organism = {
#         'dre': 7955,
#         'ath': 3702,
#         'bta': 9913,
#         'cel': 6239,
#         'dme': 7227,
#         'hsa': 9606,
#         'gga': 9031,
#         'mmu': 10090,
#         'rno': 10116,
#         'xla': 8364
#     }
#     return prefix_to_organism.get(mir.split('-')[0], 'NOT A MODDEL ORGANISM... DO WEE NEED THIS????')


def make_mir_mapping_files(mapping_path, version):
    ############################################################################
    # NEW MAPPING RULES (simpler and better!)
    # 1. use dat identifier
    #  - if the acc is in dat but the product is not, append it to the mirs
    #  - this only happens in cases like:
    #       - mmu-miR-3102-5p.2-5p instead of mmu-miR-3102-3p
    #       - and places where the product id appended .X
    # 2. use leftmost identifier:
    # to disambiguate uniques the index from the left is used as the priority.
    # if a identifier mapps to only one mir for a given priority level it is added
    #
    # OLD MAPPING RULES:
    # miRBase mapping rules:
    # For Mature MIRS
    # 1. map to the 'products' in miRNA.dat
    #    - set everything in aliases to map to that product
    #    - ignore all aliases if there is both a 5p and a 3p in the alias line
    # 2. if none use the -5p or -3p
    #   - NB THEY CAN END IN -5p.x where x is a number, usally 1
    # 3. if none exits use the longest name
    #   - if they are equal long, use the one that appears first in aliases.txt
    # NB THERE ARE INCONSISTENCIES BETWEEN THE TXT AND DAT FILES: EG
    #alias file: MIMAT0004325 -> 'ppt-miR319d.1*', 'ppt-miR319d-5p.1'
    #DAT FILE:  ppt-miR319d-3p
    # this is solved by using the dat file (ppt-miR319d-3p) as the 'map to'
    # and all alises to map_from, even though ppt-miR319d-3p is not in the alias file
    ############################################################################
    base_url = MIRBASE_BASE_URL % version

    urllib.urlretrieve("%s/aliases.txt.gz" % base_url, MIRBASE_ALIASES_TXT_PATH % version)
    urllib.urlretrieve("%s/miRNA.dat.gz" % base_url, MIRBASE_MIRNA_DAT_PATH % version)
    mat_acc_to_product = {}
    stem_acc_to_product = {}
    product_to_organism = {}
    stem_to_mat_one_to_one = set()
    organism_mapper = species_name_to_taxonomy_id()

    f_aliases_mir = open(mapping_path % version, 'w')

    acc_in_dat_file = set()
    mir_mappers = [collections.defaultdict(set) for x in range(10)]
    for line in gzip.open(MIRBASE_MIRNA_DAT_PATH % version):
        line2 = line[:2]
        if line2 == "ID":
            mat_accs, mat_products = [], []
            stem_product = line.split()[1]
        elif line2 == 'AC':
            stem_acc = line[5:].split(';')[0]
        elif line2 == 'DE':
            organism = ' '.join(line.split('stem')[0].split()[1:-1]).lower()
        elif line2 == 'FT':
            mir = re.search(r'/accession=\"([\w\d]+)\"', line)
            if mir:
                mat_acc = mir.groups()[0]
                mat_accs.append(mat_acc)
            mir = re.search(r'/product=\"(.+?)\"', line)
            if mir:
                mat_products.append(mir.groups()[0])

        elif line2 == '//' and organism in organism_mapper:
            # only save organisms that are in string!
            acc_in_dat_file.add(stem_acc)
            stem_acc_to_product[stem_acc] = stem_product
            if len(mat_products) == 1:
                p = mat_products[0]
                # then the stem can be mapped to the mature :)
                stem_to_mat_one_to_one.add(stem_acc)
                product_to_organism[stem_product] = organism_mapper[organism]
                mir_mappers[3][stem_acc].add(p)
                mir_mappers[2][stem_product].add(p)

            for p, a in zip(mat_products, mat_accs):
                acc_in_dat_file.add(a)
                product_to_organism[p] = organism_mapper[organism]
                mat_acc_to_product[a] = p
                mir_mappers[1][a].add(p)
                mir_mappers[0][p].add(p)


    # product_to_acc = {(b, a) for a, b in acc_to_product.items()}
    for line in gzip.open(MIRBASE_ALIASES_TXT_PATH % version):
        acc, products = line.rstrip(';\r\n').split('\t')
        best_product = None

        if acc in acc_in_dat_file:
            products = products.split(';')
            if acc[:5] == 'MIMAT':
                best_product = mat_acc_to_product[acc]
            elif acc in stem_to_mat_one_to_one:
                best_product = stem_acc_to_product[acc]
        if 'IMAT0000437' == acc:
           print best_product
           print products
           print acc
           print '---' * 10

        if best_product:
            used_products = {best_product, acc}  # in case it's there twice
            i = 2
            for product in products[::-1]:
                # if mir not in already_used:
                if product not in used_products:
                    mir_mappers[i][product].add(best_product)
                    i += 1
                    used_products.add(product)

    for i, mir_mapper in enumerate(mir_mappers, 1):
        for alias_mir, mirs in mir_mapper.items():
            for mir in tuple(mirs):
                organism = product_to_organism[mir]
                f_aliases_mir.write("%s\t%s\t%s\t%i\n" % ((organism), mir, alias_mir, i))



################################################################################
# STRING mapping functions:
def get_alias_mir_mapper(version=MIRBASE_VERSION):
    """Returns a dictionary that maps a mir to a list of mirs ie:
    mapping_hash[ambigious_mir] -> [unambigious_mir1, ...]
    mostly they map one-one byt if the input miR is a stem it maps to all the
    mature mirs from that stem

    Introduced mature_only flag because some matures are indiscriminate from
    their stems (no lower/upper distinction) hsa-let-7c is a stem but also a
    old name for a mature mir."""
    path = MIR_MAPPING_ALIASES_PATH

    if not os.path.exists(path % version) or DEBUGGING:
        make_mir_mapping_files(path, version)

    mapping_hash = __load_mir_mapper_alias_file__(version)
    return dict((key, sorted(values)) for (key, values) in mapping_hash.items())


def __load_mir_mapper_alias_file__(mirbase_version):
    mir_alias_path = MIR_MAPPING_ALIASES_PATH
    mapping_hash = collections.defaultdict(list)
    for line in open (mir_alias_path % mirbase_version, 'r'):
        organism, target_mir, alias_mir, priority = line.rstrip().split('\t')
        mapping_hash[alias_mir].append((int(priority), target_mir))
    return mapping_hash


def get_unique_mir_mapper(version=MIRBASE_VERSION):
    """Returns a dictionary that maps mir to mir ie:\n
    mapping_hash[ambigious_mir] -> unambigious_mir

    Introduced mature_only flag because some matures are indiscriminate from
    their stems (no lower/upper distinction) hsa-let-7c is a stem but also a
    old name for a mature mir."""
    mapper = {}
    alias_mapper = get_alias_mir_mapper(version)
    for from_mir, to_mirs in alias_mapper.items():
        if len(to_mirs) == 1:
            mapper[from_mir] = to_mirs[0][1]
    return mapper


def get_mir_id_to_tax_id_mapper(mirbase_version=MIRBASE_VERSION):
    """

    :param mirbase_version: miRBase version to be used
    :return: dict string -> string, mapping RNA identifiers to their respective taxonomy identifiers
    """
    mir_alias_path = MIR_MAPPING_ALIASES_PATH
    mir_id_to_tax_id = {}
    for line in open(mir_alias_path % mirbase_version, 'r'):
        organism, target_mir, alias_mir, priority = line.rstrip().split('\t')
        mir_id_to_tax_id[target_mir] = organism
    return mir_id_to_tax_id


################################################################################

def get_string_mapping_file(organisms, filter_string_id, filter_string_alias,
                            version, db, include_basename):

    assert 10 >= version >= 8, 'only version 8, 9 and 10 are supported'
    if version == 10:
        string_aliases = STRING_ALIASES_100
    elif version == 9:
        string_aliases = STRING_ALIASES_91
    elif version == 8:
        string_aliases = STRING_ALIASES_83
    else:
        raise ValueError('Unknown STRING version: {:d}'.format(version))

    class Everything(object):
        def __contains__(self, item):
            return True
        def __iter__(self):
            yield 'all'
        def __len__(self):
            return np.inf

    def parse_arg(arg):
        if arg == 'all':
            return Everything()
        elif isinstance(arg, (int, str)):
            return {str(arg)}
        elif isinstance(arg, (list, tuple, set)):
            return set(map(str, arg))

    db = parse_arg(db)
    organisms = parse_arg(organisms)

    string_aliases_file = os.path.join(DATA_PATH, string_aliases.split('/')[-1])

    # debugging speed hack
    # if version == 9:
    #     string_aliases_file = 'test.txt.gz'

    if not os.path.exists(string_aliases_file) or DEBUGGING:
        urllib.urlretrieve(string_aliases, string_aliases_file)

    args = '_'.join(('+'.join(sorted([str(x) for x in organisms])), filter_string_alias,
                     filter_string_id, str(version), '+'.join(db), str(include_basename)[0]))

    mapping_file = os.path.join(ID_PATH, 'string_%s.tsv' % args )

    l_filter_id, l_filter_alias = len(filter_string_id), len(filter_string_alias)
    base_names = []
    used_names = set()
    if not os.path.exists(mapping_file) or DEBUGGING:
        with open(mapping_file, 'w') as new_mapping_file:
            for line in gzip.open(string_aliases_file):
                if line[0] == '#':
                    continue
                if version == 10:
                    organism, string_alias, _db = line.rstrip().split('\t')
                    organism = organism.split(".")
                    string_id = ".".join(organism[1:])
                    organism = organism[0]
                else:
                    organism, string_id, string_alias, _db = line.rstrip().split('\t')
                if organism in organisms and filter_string_id == string_id[:l_filter_id] \
                        and filter_string_alias == string_alias[:l_filter_alias] and _db in db:
                    new_mapping_file.write('%s\t%s\t%s\n' % (organism, string_alias, string_id))
                    if include_basename:
                        used_names.add(string_alias)
                        used_names.add(string_id)
                        if '.' in string_alias:
                            base_names.append((organism, string_alias, string_id))

            # chomps of names ending with .x eg NM_3131232.2 becomes NM_3131232
            # if this name dose't map to something else


            for organism, string_alias, string_id in base_names:
                # error proteins.tsv.gz has the following line ?!?!?!
                #10090	ENSMUSP00000065966	.	BLAST_KEGG_NAME Ensembl_EntrezGene_synonym
                if string_alias == '.':
                    continue
                index = 0
                for i in range(string_alias.count('.')):
                    #  print i, string_alias, string_alias[:index]
                    index = string_alias.index('.', index+1)
                    new_string_alias = string_alias[:index]
                    if new_string_alias not in used_names:
                        new_mapping_file.write('%s\t%s\t%s\n' % (organism, new_string_alias, string_id))
                        used_names.add(new_string_alias)

    return mapping_file


# ensp8_to_ensg = stringrnautils.get_string_to_alias_mapper(9606, 'ENSP', 'ENSG', 8)['9606']
def get_string_to_alias_mapper(organisms="9606", filter_string_id='ENSP',
                               filter_string_alias='ENSG', version=10, db='all', include_basename=True):
    """parses the string alias files, and generates a mapper of mapper[organism][string_id] = string_alias
     - note that all keys in the mapper are strings
    if organisms = 'all', every organism is added to the mapper,
    if organisms = list, tuple or set only thouse will be avalible in the mapper
    if organisms = a integer only that organism will be avalible in the mapper

    db is an optional argument if you want to filter based based on the last last column of the string alias file
     - default behavious is to ignore this coloumn (by setting it to all)

    to_id and from_id are supstrings that have to be found in the corresponding collums to be put in the mapper"""

    mapping_file = get_string_mapping_file(organisms, filter_string_id,
                                           filter_string_alias, version, db, include_basename)

    mapper = collections.defaultdict(dict)
    for line in open(mapping_file):
        organism, string_alias, string_id = line.rstrip().split('\t')
        mapper[organism][string_id] = string_alias

    return dict(mapper)

def get_alias_to_string_mapper(organisms="9606", filter_string_id='ENSP',
                               filter_string_alias='ENSG', version=10, db='all', include_basename=True):
    """parses the string alias files, and generates a mapper of mapper[organism][string_alias] = string_id
     - note that all keys in the mapper are strings
    if organisms = 'all', every organism is added to the mapper,
    if organisms = list, tuple or set only thouse will be avalible in the mapper
    if organisms = a integer only that organism will be avalible in the mapper

    db is an optional argument if you want to filter based based on the last last column of the string alias file
     - default behavious is to ignore this coloumn (by setting it to all)

    to_id and from_id are supstrings that have to be found in the corresponding collums to be put in the mapper"""
    mapping_file = get_string_mapping_file(organisms, filter_string_id,
                                           filter_string_alias, version, db, include_basename)

    mapper = collections.defaultdict(dict)
    for line in open(mapping_file):
        organism, string_alias, string_id = line.rstrip().split('\t')
        mapper[organism][string_alias] = string_id

    return dict(mapper)

#########################################################################
### RNA mapping functions
########################################################################


def get_non_coding_rna_alias_mapper():
    """
    Generates a dictionary mapping ncRNA aliases in different organisms to the corresponding RAIN ncRNA identifier,

    :return: a dictionary (str -> str -> str): taxonomy ID -> RNA alias -> RNA identifier
    """
    ncrna_url = "http://gdurl.com/qA1t/download"
    ncrna_file = os.path.join(ID_PATH, "ncRNAaliasfile.tsv.gz")

    if not os.path.exists(ncrna_file):
        os.system("wget --no-check-certificate -q -O %s '%s'" % (ncrna_file, ncrna_url))

    handle = gzip.open(ncrna_file) if ncrna_file.endswith(".gz") else open(ncrna_file)
    tmp_list = [x.strip("\n").split("\t") for x in handle]
    handle.close()
    ncrna_mapper = collections.defaultdict(dict)
    for tax, identifier, alias, source in tmp_list:
        ncrna_mapper[tax][alias] = identifier
    return ncrna_mapper


def get_rna_identifiers_in_organism(rna_aliases_file):
    """

    :param rna_aliases_file: RNA alias file as created by script create_rna_aliases_file.py
    :return: a dictionary: taxonomy ID -> RNA identifiers
    """
    mapper = collections.defaultdict(set)
    with gzip.open(rna_aliases_file, 'rb') as rna_file:
        # skip header
        next(rna_file)
        for line in rna_file:
            tax, rna_id, rna_alias, sources_string = line.rstrip('\n\r').split('\t')
            mapper[tax].add(rna_id)
    return mapper


################################################################################
#   NPINTER mappers
################################################################################
######### UniProt to STRINGidentifier
def getUniProtDic(archivepath):
    idDic = {'ce': '6239', 'dr': '7955', 'dm': '7227', 'hs': '9606', 'mm': '10090', 'oc': '9986', 'sc': '4932'}
    archive = zipfile.ZipFile(archivepath, 'r')
    Uniprot_dic = {}

    for org in idDic.keys():
        uniprot_path = org + '_ENSG_UniProt.tsv'
        uniprot_f = archive.open(uniprot_path, 'r')
        uniprot_f.readline()
        Uniprot_dic[idDic[org]] = {}

        STRING_dic = get_alias_to_string_mapper(organisms=idDic[org], filter_string_alias='', filter_string_id='')[
            idDic[org]]

        for line in uniprot_f:
            cols = line.rstrip().split("\t")
            if STRING_dic.has_key(cols[0]):
                ensemblid = STRING_dic[cols[0]]
                if len(cols) > 1 and len(cols[1]) > 0:
                    Uniprot_dic[idDic[org]][cols[1]] = ensemblid
                if len(cols) > 2 and len(cols[2]) > 0:
                    Uniprot_dic[idDic[org]][cols[2]] = ensemblid

            elif len(cols) > 1 and STRING_dic.has_key(cols[1]):
                Uniprot_dic[idDic[org]][cols[1]] = STRING_dic[cols[1]]

            elif len(cols) > 2 and STRING_dic.has_key(cols[2]):
                Uniprot_dic[idDic[org]][cols[2]] = STRING_dic[cols[2]]

        uniprot_f.close()
    return Uniprot_dic

######### RefSeq(NM) to STRING identifier
#RefSeq (NM_ mRNA) to ENSP
def getRefSeqNMdic(archivepath):
    idDic={'ce': '6239','dr':'7955','dm':'7227','hs': '9606','mm':'10090','oc':'9986','sc':'4932'}
    archive = zipfile.ZipFile(archivepath, 'r')
    RefSeq_mRNA_dic = {}
    for org in idDic.keys():
        NM_path = org+'_ENSG_RefSeq_mRNA.tsv'
        NM_f = archive.open(NM_path,'r')
        NM_f.readline()

        STRING_dic = get_alias_to_string_mapper(organisms=idDic[org],filter_string_alias='', filter_string_id='')[idDic[org]]
        RefSeq_mRNA_dic[idDic[org]] = {}

        for line in NM_f:
            cols  = line.rstrip().split("\t")
            if len(cols)>1:
                if STRING_dic.has_key(cols[0]):
                    ensemblid = STRING_dic[cols[0]]
                    if len(cols[1])>0:
                        RefSeq_mRNA_dic[idDic[org]][cols[1]]=ensemblid
                elif STRING_dic.has_key(cols[1]):
                    RefSeq_mRNA_dic[idDic[org]][cols[1]]=STRING_dic[cols[1]]
        NM_f.close()

    return RefSeq_mRNA_dic

############ EnsemblID to GeneName
def getncRNAtoGeneNamedic(GeneNamePath):
    toGene_dic = {}
    with gzip.open(GeneNamePath,'r') as genname_f:
        for line in genname_f:
            cols  = line.rstrip().split("\t")
            if not toGene_dic.has_key(cols[0]):
                toGene_dic[cols[0]]={}
            for geneid in cols[2].split(';'):
                toGene_dic[cols[0]][geneid]=cols[1]
    return toGene_dic

############# RefSeq (NR_ ncRNA) to GeneName (HGNC,FlyBase,?) (ENSEMBL conversion in between)
def getRefSeqNRdic(archivepath,GeneNamePath):
    idDic={'ce': '6239','dr':'7955','dm':'7227','hs': '9606','mm':'10090','oc':'9986','sc':'4932'}
    archive = zipfile.ZipFile(archivepath, 'r')
    RefSeq_ncRNA_dic = {}
    GeneDic = getncRNAtoGeneNamedic(GeneNamePath)

    for org in idDic.keys():
        NR_path = org+'_ENSG_RefSeq_ncRNA.tsv'
        NR_f = archive.open(NR_path,'r')
        NR_f.readline()

        RefSeq_ncRNA_dic[idDic[org]] = {}
        if GeneDic.has_key(idDic[org]):
            toGene_dic = GeneDic[idDic[org]]

            for line in NR_f:
                cols  = line.rstrip().split("\t")
                if len(cols)>1:
                    if len(cols[1])>0:
                        if toGene_dic.has_key(cols[0]):
                            RefSeq_ncRNA_dic[idDic[org]][cols[1]]=toGene_dic[cols[0]]
        NR_f.close()

    return RefSeq_ncRNA_dic

################ NONCODE 2 GeneName (NCv3->NCv4->(ENST or RefSeqncRNA(NR_))->ENSG->Genename)
def getNONCODEdic(noncode_path,archivepath,GeneNamePath):
    GeneDic = getncRNAtoGeneNamedic(GeneNamePath)
    RefSeq_ncRNA_dic = getRefSeqNRdic(archivepath,GeneNamePath)
    ENST_dic = {}
    NONCODE_dic = {}

    idDic={'ce': '6239','dr':'7955','dm':'7227','hs': '9606','mm':'10090','oc':'9986','sc':'4932'}
    archive = zipfile.ZipFile(archivepath, 'r')
    for org in idDic.keys():
        ENSTpath = org+'_ENSG_ENST.tsv'
        ENST_f = archive.open(ENSTpath,'r')
        ENST_dic[idDic[org]] = {}
        NONCODE_dic[idDic[org]] = {}
        ENST_f.readline()
        if GeneDic.has_key(idDic[org]):
            toGene_dic = GeneDic[idDic[org]]
            for line in ENST_f:
                cols  = line.rstrip().split("\t")
                if len(cols)>1:
                    if len(cols[1])>0:
                        if toGene_dic.has_key(cols[0]):
                            ENST_dic[idDic[org]][cols[1]]=toGene_dic[cols[0]]
        ENST_f.close()

    #v4_f = open(noncodev4_path,'r')
    #v4_dic = {}
    #for line in v4_f:
    #	cols = line.strip().split('\t')
    #	if len(cols)==2:
    #		v4_dic[cols[1]]=cols[0]
    #v4_f.close()

    noncode_f = open(noncode_path,'r')
    for line in noncode_f:
        cols  = line.strip().split('\t')
        NCid = cols[0]
        found = False
        for i in range(1,len(cols)):
            if not cols[i].startswith("NA"):
                for orgid,RefSeq_dic in RefSeq_ncRNA_dic.items():
                    if (not found) and RefSeq_dic.has_key(cols[i]):
                        NONCODE_dic[orgid][NCid] = RefSeq_dic[cols[i]]
                        found=True
        if not found:
            for i in range(1,len(cols)):
                if not cols[i].startswith("NA"):
                    for orgid,ENST in ENST_dic.items():
                        if(ENST.has_key(cols[i])):
                            NONCODE_dic[orgid][NCid] = ENST[cols[i]]
                            found=True
    noncode_f.close()
    return NONCODE_dic

def getAliasFORncRNAs(noncode_path,archivepath,GeneNamePath,outputfile):
    dics = [getncRNAtoGeneNamedic(GeneNamePath),getRefSeqNRdic(archivepath,GeneNamePath),getNONCODEdic(noncode_path,archivepath,GeneNamePath)]
    with open(outputfile,'w') as out_f:
        for dic in dics:
            for org,aliasdic in dic.items():
                for old,new in aliasdic.items():
                    out_f.write('\t'.join([org,new,old])+'\n')
    print 'DONE'
################################################################################
#  benchmark functions
################################################################################

def get_prior(gold_standard_file, data_set=None, use_blacklist=False, expanison = False):

    ref_data, ref_rnas, ref_prots = parse_marster_file(gold_standard_file, expanison = expanison)
    if data_set:
        data, rnas, prots = parse_marster_file(data_set, expanison = expanison)
    else:
        data, rnas, prots = ref_data, ref_rnas, ref_prots
    
    if use_blacklist:
        blacklist = get_benchmark_blacklist(ref_data)
        rnas, prots, scores = filter_interactions(rnas, prots, blacklist)

    common_rnas = get_common(rnas, ref_rnas)
    common_prots = get_common(prots, ref_prots)

    set_id1, set_id2 = set(), set()
    n_true=0
    for rna_id, prot_id in ref_data['9606'].keys():
        if rna_id in common_rnas and prot_id in common_prots:
            set_id1.add(rna_id)
            set_id2.add(prot_id)
            n_true += 1

    n_false = len(set_id1) * len(set_id2)
    return SCALING_FACTOR * float(n_true) / n_false


def fit_to_sigmoid(raw_scores, true_pos_rate, increases=True, fit_name='test',
                   window_size=25, max_value=0.9, ignore_fraction=0.0):

    l = len(raw_scores)
    n_ignore = int(round(l * ignore_fraction))

    true_pos_rate = np.array(true_pos_rate, dtype=float)
    raw_scores = np.array(raw_scores)
    jitter = (np.random.random(len(raw_scores)) - 0.5) * (0.3 / window_size)

    a = min(true_pos_rate.max(), max_value)
    d = true_pos_rate.min()

    c_targets = ((3 * a + d) / 4.0, (a + d) / 2.0, (a + 3 * d) / 4.0)
    c_indexes = [sum([x < ct for x in true_pos_rate]) for ct in c_targets]
    b_div = 1.0 * abs(min(raw_scores)-max(raw_scores))
    #bs = [1 / b_div, 10 / b_div, 100 / b_div]
    bs = [1 / b_div, 10 / b_div, 30 / b_div]
    b_limit = bs[2]

    if increases:
        # bs = [lmfit.Parameter(value = b, min = b_limit) for b in bs]
        bs = [lmfit.Parameter(value = b, min = 0) for b in bs]
        cs = [lmfit.Parameter(value = raw_scores[c_index]) for c_index in c_indexes]
    else:
        # bs = [lmfit.Parameter(value = -b, max = -b_limit) for b in bs]
        bs = [lmfit.Parameter(value = -b, max = 0) for b in bs]
        cs = [lmfit.Parameter(value = raw_scores[-c_index]) for c_index in c_indexes]
        true_pos_rate = true_pos_rate[::-1]
        raw_scores = raw_scores[::-1]

    jittered_true_pos_rate = true_pos_rate + jitter

    # was a good idear if TP and FP was more balanced
    #_mean = true_pos_rate.sum() / len(true_pos_rate)
    # this only works with large windows
    _mean = (a + d) / 2
    a = lmfit.Parameter(value=a, min=_mean, max=a)
    d = lmfit.Parameter(value=d, min=0, max=_mean)
    bscs = [(b, c) for b in bs for c in cs]
    fits = []


    ini_axes = plt.figure().add_subplot(1,1,1)
    ini_axes.plot(raw_scores, jittered_true_pos_rate, 'r.')

    x = np.linspace(raw_scores.min(), raw_scores.max(), 1000)

    # def sigmoid_fun(x, a, b, c, d):
    #     return (a - d) / (1 + np.exp(-1 * b * (x - c))) + d

    l_ = l / 10
    _points = [p for i, p in enumerate(raw_scores) if i % l_ == 0]
    ini_axes.plot(_points, [0.1] * len(_points), 'kx')
    #to_args = lambda p : [p[x].value for x in 'abcd']
    # import pdb; pdb.set_trace()
    for (b, c) in bscs:
        ini_axes.plot(x, sigmoid_fun(x, a.value, b.value, c.value, d.value), 'b-')

        model = lmfit.Model(sigmoid_fun, independent_vars=['x'])
        result = model.fit(true_pos_rate[n_ignore:], x=raw_scores[n_ignore:],
                           # _files = ("m2000.tsv", 's2000.tsv')#, "starmirdb.tsv", "miRanda.tsv", "TargetScan_40.tsv")
                           a = a, b = b, c = c, d = d)
        fits.append((result.chisqr, result.values))
    plt.savefig(os.path.join(FIG_PATH, '%s_initial_guesses.eps' % fit_name), format='eps')

    fits.sort()
    final_axes = plt.figure().add_subplot(1,1,1)
    final_axes.plot(raw_scores[n_ignore:], jittered_true_pos_rate[n_ignore:], 'r.')
    final_axes.plot(raw_scores[:n_ignore], jittered_true_pos_rate[:n_ignore], 'k.')
    for chisqr, params in fits:
        final_axes.plot(x, sigmoid_fun(x, **params), 'y-')
    plt.savefig(os.path.join(FIG_PATH, '%s_all_fits.eps' % fit_name), format='eps')

    best_result = fits[0][1]
    best_fit_axes = plt.figure().add_subplot(1,1,1)
    plt.xlabel('Raw score')
    plt.ylabel('Probability')
    best_fit_axes.plot(raw_scores[n_ignore:], jittered_true_pos_rate[n_ignore:], 'r.')
    best_fit_axes.plot(raw_scores[:n_ignore], jittered_true_pos_rate[:n_ignore], 'k.')
    best_fit_axes.plot(x, sigmoid_fun(x, **best_result), 'g-')
    plt.savefig(os.path.join(FIG_PATH, '%s_best_fit.eps' % fit_name), format='eps')
    #plt.show()
    sys.stderr.write("Parameters for sigmoid: " + str(best_result) + "\n")
    #import pdb
    #pdb.set_trace()

    return lambda x : sigmoid_fun(x, **best_result)
    # return parameters


def score_plot(scores, true_pos_rate, fun, fit_name, parameters):

    scores = np.array(scores)
    true_pos_rate = np.array(true_pos_rate, dtype=float)

    fig = plt.figure()
    axes = fig.add_subplot(1,1,1)
    axes.plot(scores, true_pos_rate, 'r.')

    x = np.linspace(min(scores), max(scores), 100)
    axes.plot(x, fun(x, **parameters), 'b-')
    # plt.show()
    plt.savefig(os.path.join(FIG_PATH, fit_name) )

def read_ncRNA_alias():
    ncRNA_alias_identifier_map ={}
    fp = gzip.open('/home/panxy/eclipse/string-rna/id_dictionaries/ncRNAaliasfile.tsv.gz')  
    for line in fp:
        species, identity, alias, source = line.rstrip('\r\n').split('\t')
        if '9606' != species:
            continue
        else:
            ncRNA_alias_identifier_map[alias.upper()] = identity
    fp.close()         

    fp1 = gzip.open('/home/panxy/eclipse/string-rna/id_dictionaries/ncRNAorthfile.tsv.gz')  
    for line in fp1:
        values = line.rstrip('\r\n').split('\t')
        if ncRNA_alias_identifier_map.has_key(values[1].upper()):
            continue
        ncRNA_alias_identifier_map[values[1].upper()] = values[0]
    fp1.close()   
    
    ncRNA_alias_identifier_map['ENSG00000281560'] = 'LSINCT5'
    ncRNA_alias_identifier_map['LSINCT5'] = 'LSINCT5'
    
    ncRNA_alias_identifier_map['ENSG00000274006'] = 'DLG2AS'
    ncRNA_alias_identifier_map['DLG2AS'] = 'DLG2AS'
    
    ncRNA_alias_identifier_map['ENSG00000280989'] = 'BX118339'
    ncRNA_alias_identifier_map['BX118339'] = 'BX118339'
    
    ncRNA_alias_identifier_map['ENSG00000281344'] = 'HELLPAR'
    ncRNA_alias_identifier_map['HELLPAR'] = 'HELLPAR'
    
    ncRNA_alias_identifier_map['ENSG00000280752'] = 'KUCG1'
    ncRNA_alias_identifier_map['KUCG1'] = 'KUCG1'
    
    ncRNA_alias_identifier_map['ENSG00000255090'] = 'MIR100HG'
    ncRNA_alias_identifier_map['MIR100HG'] = 'MIR100HG'    
    ncRNA_alias_identifier_map['ENSG00000260978'] = 'MKRN3-AS1'
    ncRNA_alias_identifier_map['MKRN3-AS1'] = 'MKRN3-AS1'  

    ncRNA_alias_identifier_map['ENSG00000281183'] = 'NPTN-IT1'
    ncRNA_alias_identifier_map['NPTN-IT1'] = 'NPTN-IT1'     

    ncRNA_alias_identifier_map['ENSG00000281450'] = 'PANDAR'
    ncRNA_alias_identifier_map['PANDAR'] = 'PANDAR'
    ncRNA_alias_identifier_map['ENSG00000281398'] = 'SNHG4'
    ncRNA_alias_identifier_map['SNHG4'] = 'SNHG4'    
    ncRNA_alias_identifier_map['ENSG00000281881'] = 'SPRY4-IT1'
    ncRNA_alias_identifier_map['SPRY4-IT1'] = 'SPRY4-IT1'                  
    ncRNA_alias_identifier_map['ENSG00000281664'] = 'YIYA'
    ncRNA_alias_identifier_map['YIYA'] = 'YIYA'

    ncRNA_alias_identifier_map['ENSG00000281406'] = 'BLACAT1'
    ncRNA_alias_identifier_map['BLACAT1'] = 'BLACAT1'    
      
    '''fp2 = gzip.open('data/dict/RNA_long_non-coding.txt.gz')  
    for line in fp2:
        if 'Synonyms' in line:
            continue
        values = line.rstrip('\r\n').split('\t')
        if ncRNA_alias_identifier_map.has_key(values[1]):
            continue
        ncRNA_alias_identifier_map[values[1]] = values[1]
        synoms = values[4].split(',')
        for synom in synoms:
            if ncRNA_alias_identifier_map.has_key(values[1]):
                continue  
            ncRNA_alias_identifier_map[synom] = values[1]  
         
        if  ncRNA_alias_identifier_map.has_key(values[-2]):
            continue
        else:
            ncRNA_alias_identifier_map[values[-2]] = values[1]   
              
    fp2.close()        
    '''   
    return ncRNA_alias_identifier_map

def read_disease_files(file_name):
    disease_genes = {}
    for line in open(file_name, 'r'):
        values = line.rstrip('\r\n').split('\t')
        gene = values[1]
        disease = values[2]
        disease_genes.setdefault(disease, set([])).add(gene)
    return disease_genes
        
def parse_marster_file(file_name, expanison = False):
    # NOTE ONLY BENCHMARKS ON HUMAN, this is important if we ever change gold standard!!!!
    # NOTE ONLY BENCHMARKS ON HUMAN, this is important if we ever change gold standard!!!!
    # NOTE ONLY BENCHMARKS ON HUMAN, this is important if we ever change gold standard!!!!
    # NOTE ONLY BENCHMARKS ON HUMAN, this is important if we ever change gold standard!!!!
    # NOTE ONLY BENCHMARKS ON HUMAN, this is important if we ever change gold standard!!!!
    # NOTE ONLY BENCHMARKS ON HUMAN, this is important if we ever change gold standard!!!!
    sys.stderr.write("- Reading file " + str(file_name) + "\n")
    disease_genes = read_disease_files(file_name)
    oboparser = obo_object()
    oboparser.read_obo_file()
    data = {}
    id1s = {}
    id2s = {}
    for line in open(file_name):
        values = line.rstrip('\r\n').split('\t')
        organism = values[0]
        '''gene = values[1].upper()
        id1 = ''
        if ncRNA_alias_identifier_map.has_key(gene):
            id1 = ncRNA_alias_identifier_map[gene]
        else:
            gene_alias = values[10].split(';')
            for val_name in gene_alias:
                if ncRNA_alias_identifier_map.has_key(val_name):
                    id1 = ncRNA_alias_identifier_map[val_name]
                    break
                else:           
                    print 'no map ensp', gene
                    continue
        if id1 == '':
            continue
        '''
        id1 = values[1]
        #pdb.set_trace()
        #if 'ENSG' not in id1:
        #    continue
        id2 = values[2]
            
        score = values[5]                     
                
        if(organism == "9606"):
            if(not data.has_key(organism)):
                data[organism] = {}
            if(data[organism].has_key((id1, id2))):
                if(data[organism][(id1, id2)] <= float(score)):
                    data[organism][(id1, id2)] = float(score)
            else:
                data[organism].update({(id1, id2) : float(score)})
            id1s[id1] = 1
            id2s[id2] = 1
        if expanison:
            if 'DOID' in id2:
                child_disease = oboparser.getDescendents(id2)
                for dis in child_disease:
                    if disease_genes.has_key(dis):
                        genes = disease_genes[dis]
                        for id1_exp in genes:
                            if(organism == "9606"):
                                if(not data.has_key(organism)):
                                    data[organism] = {}
                                if(data[organism].has_key((id1_exp, id2))):
                                    if(data[organism][(id1_exp, id2)] <= float(score)):
                                        data[organism][(id1_exp, id2)] = float(score)
                                else:
                                    data[organism].update({(id1_exp, id2) : float(score)})
                                id1s[id1_exp] = 1
                                id2s[id2] = 1 
                                
                if disease_genes.has_key(id2):
                    own_genes = disease_genes[id2]  
                if 'DOID' in id2:
                    parent_disease = oboparser.getAncestors(id2)
                    for dis in parent_disease:
                        for id1_exp in own_genes:
                            if(organism == "9606"):
                                if(not data.has_key(organism)):
                                    data[organism] = {}
                                if(data[organism].has_key((id1_exp, dis))):
                                    if(data[organism][(id1_exp, dis)] <= float(score)):
                                        data[organism][(id1_exp, dis)] = float(score)
                                else:
                                    data[organism].update({(id1_exp, dis) : float(score)})
                                id1s[id1_exp] = 1
                                id2s[dis] = 1                              
    return(data,id1s,id2s)


def get_common(id_1, id_2):
    common = set(id_1.keys()).intersection(set(id_2))
    return(common)



def get_benchmark_blacklist(gold_data):
    """
    filters negative set by removeing all mirs that share family with the positive set
    we do this because these are potentially correct but undiscovered, thus they are 'grey listed'
    this information is parsed from the textmining files
        -----  WARNING FOR FUTURE DEVELOPERS -----
            ### we only have HUMAN (no other organisms) in our gold standard
            ### We only have miR (no other rnas) in our gold standard
            ### if this is changed this script needs to be rewritten!
            ### because the ids it is called with have no 'organism' information, so human is assumed!
        ----- END WARNING -----
    """
    print gold_data.keys()
    print gold_data["9606"]
    positive_set = set(gold_data["9606"].keys())

    black_list = set()

    # entity_to_serial = {}
    serial_to_entity = {}
    for line in open(os.path.join(ID_PATH, 'miR_entityfile.tsv')):
        id_serial, id_organism, id_mir = line.rstrip().split('\t')
        # entity_to_serial[id_mir] = id_serial
        serial_to_entity[id_serial] = id_mir.split(':')[-1]

    mir_groupfile = open(os.path.join(ID_PATH, 'miR_groupfile.tsv'))
    group_to_entities = collections.defaultdict(set)
    entiry_to_group = {}
    for line  in mir_groupfile:
        group_serial, id_serial = line.rstrip().split('\t')
        id_mir = serial_to_entity[id_serial].split(':')[-1]
        entiry_to_group[id_mir] = group_serial
        group_to_entities[group_serial].add(id_mir)
        
    for positive_mir, positive_protein in positive_set:
        mir_family = entiry_to_group[positive_mir]
        for mir in group_to_entities[mir_family]:
            pair = (mir, positive_protein)
            if pair not in positive_set:
                black_list.add(pair)

    return black_list


def filter_interactions(rna_ids, protein_ids, blacklist):
    """
    :param rna_ids: a list of rna ids
    :param protein_ids: a list of protein ids
    :param blacklist: a set of (rna_id, protein_id) to be excluded
    :return (rna_ids, protein_ids) that are not in blacklist
    """
    #else:
    new_rna_ids, new_protein_ids= [] ,[]
    for rna, protein in zip(rna_ids, protein_ids):
        if (rna, protein) not in blacklist:
            new_rna_ids.append(rna)
            new_protein_ids.append(protein)
    return new_rna_ids, new_protein_ids 


def filter_data(rna_ids, protein_ids, scores, blacklist):
    """
    :param rna_ids: a list of rna ids
    :param protein_ids: a list of protein ids
    :param scores: a list of scores
    :param blacklist: a set of (rna_id, protein_id) to be excluded
    :return (rna_ids, protein_ids, scores) that are not in blacklist
    """
    #else:
    new_rna_ids, new_protein_ids, new_scores = [], [] ,[]
    for rna, protein, score in zip(rna_ids, protein_ids, scores):
        if (rna, protein) not in blacklist:
            new_rna_ids.append(rna)
            new_protein_ids.append(protein)
            new_scores.append(score)
    return new_rna_ids, new_protein_ids, new_scores



def discrete_benchmark(organism_ids, rna_ids, protein_ids, assigned_bins, gold_standard_file,
                       out_file_name='test', use_blacklist =False, expanison = False):
    """
    Computes confidence for a set of interactions where each interactions is assigned to one or several bins.
    The confidence of each bin is the precision with respect to the gold standard but restricted to RNAs and proteins
    that also occur in the given gold standard set of interactions.
    Finally, the confidence of an interaction is the maximum confidence of all bins it is assigned to.

    :param organism_ids: collection of strings - taxonomy identifiers of the organism where the interaction was observed
    :param rna_ids: collection of strings - identifiers of the interacting RNAs
    :param protein_ids: collection of strings - identifiers of the interacting proteins
    :param assigned_bins: collection of collections of strings - the bins each interaction is assigned to
    :param gold_standard_file: string - name of the gold standard file to be used for scoring
    :param out_file_name: name of the output file, a diagnostic output is written to
    :return: list of float - the confidence of each interaction or nan if no confidence could be computed
    """
    
    gold_data, gold_rnas, gold_prots = parse_marster_file(gold_standard_file, expanison = expanison )
    if use_blacklist:
        blacklist = get_benchmark_blacklist(gold_data)
        rna_ids, protein_ids, scores = filter_data(rna_ids, protein_ids, assigned_bins, blacklist)

    # Maps each pair of interacting RNAs and proteins to a list of bins assigned to this interaction
    interactions_to_bins = {}
    # Maps each bin to the number of occurrences in this data set
    bin_to_occurrences = collections.defaultdict(int)
    for org, rna, protein, bins in zip(organism_ids, rna_ids, protein_ids, assigned_bins):
        # Make sure that all assigned bins are a list, tuple or set
        if not isinstance(bins, (list, tuple, set)):
            bins_collection = [bins]
        else:
            bins_collection = bins
        for b in bins_collection:
            bin_to_occurrences[b] += 1
        if org not in interactions_to_bins:
            interactions_to_bins[org] = {(rna, protein): bins_collection}
        else:
            interactions_to_bins[org][rna, protein] = bins_collection

    # Returns a dict of interactions as described above for the gold standard and two additional dicts for the RNA and
    # protein identifiers that simply map the identifiers to themselves

    common_rnas = get_common(gold_rnas, rna_ids)
    sys.stderr.write("The number of common ncRNAs is: " + str(len(common_rnas)) + "\n")
    common_proteins = get_common(gold_prots, protein_ids)
    sys.stderr.write("The number of common proteins is: " + str(len(common_proteins)) + "\n")

    sys.stderr.write("Started benchmarking the data set\n")
    positive = 1
    negative = 0
    # Vector of two-element tuples of the form (this_bin, 0 or 1) where the second element is 1 if a TP interaction was found
    # for the respective this_bin and 0 if a FP interaction was seen
    vector = []
    positives = 0
    negatives = 0
    for org in interactions_to_bins.keys():
        for rna, protein in interactions_to_bins[org].keys():
            bins = interactions_to_bins[org][(rna, protein)]
            if (rna in common_rnas) and (protein in common_proteins):
                for curr_bin in bins:
                    if (rna, protein) in gold_data[org]:
                        vector.append((curr_bin, positive))
                        positives += 1
                    else:
                        vector.append((curr_bin, negative))
                        negatives += 1
    vector.sort(key= lambda x: x[0])

    # Map each bin to the number of TP and the number of (TP+FP)
    bin_to_tps = collections.defaultdict(int)
    bin_to_total = collections.defaultdict(int)
    for bin_name, pos_or_neg in vector:
        bin_to_tps[bin_name] += pos_or_neg
        bin_to_total[bin_name] += 1

    bin_to_confidence = {}
    for bin_name, tps in bin_to_tps.items():
        tps *= SCALING_FACTOR
        total = bin_to_total[bin_name]
        bin_to_confidence[bin_name] = min(tps / float(total), 0.9)  # Highest possible confidence is 0.9

    interaction_confidences = []
    for bins in assigned_bins:
        if not isinstance(bins, (list, tuple, set)):
            bins_collection = [bins]
        else:
            bins_collection = bins
        max_conf = float('-inf')
        for curr_bin in bins_collection:
            if curr_bin in bin_to_confidence:
                curr_conf = bin_to_confidence[curr_bin]
                if curr_conf > max_conf:
                    max_conf = curr_conf
        if max_conf == float('-inf'):
            max_conf = float('nan')
        interaction_confidences.append(max_conf)

    # Print confidences to file and stderr for diagnosis
    out_file_name_full = out_file_name + '.txt'
    with open(out_file_name_full, 'w') as f_out:
        f_out.write('\t'.join(("Assay", "Occurrences", "TP", "TP+FP", "Precision")) + "\n")
        for this_bin in sorted(bin_to_occurrences.keys()):
            bin_occurrences = bin_to_occurrences[this_bin]
            if this_bin in bin_to_confidence:
                tps = bin_to_tps[this_bin]
                tot = bin_to_total[this_bin]
                conf = bin_to_confidence[this_bin]
            else:
                tps = 0
                tot = 0
                conf = float('nan')
            f_out.write("\t".join((str(this_bin), str(bin_occurrences), str(tps), str(tot), str(conf)))+ "\n")
    for line in open(out_file_name_full, 'r'):
        sys.stderr.write(line)

    sys.stderr.write("Finished benchmarking the data set\n")
    return interaction_confidences




def benchmark(organisms, rna_ids, protein_ids, scores, gold_standard_file=GOLDSTANDARD_FILE,
              increases=True, window_size=100, fit_name='test', discrete=False, max_value=0.9,
              ignore_fraction=0.0, use_blacklist=False, expanison = False):
    """needs 4 args: organism, rna_ids and protein_ids, scores are vectors
       optional args: debugging True (plots the fits twice - first initial guess, then final fit)
       windows_size used to estimate the overlap between scores and golden standard
       increases = True if 'higher scores' are better, otherwise false
       goldstandard_file, file in master format to benchmark against, default=croft

         - returns list of 'confidence scores' """

    if discrete:
        return discrete_benchmark(organisms, rna_ids, protein_ids, scores,
                                  gold_standard_file,
                                  out_file_name=fit_name, use_blacklist=use_blacklist)

    print gold_standard_file
    gold_data, gold_rnas, gold_prots = parse_marster_file(gold_standard_file, expanison = expanison)
    if use_blacklist:
        blacklist = get_benchmark_blacklist(gold_data)
        rna_ids, protein_ids, scores = filter_data(rna_ids, protein_ids, scores, blacklist)

    # Continuous scoring
    original_scores = scores
    data = {}
    data_rnas, data_prots = rna_ids, protein_ids
    for org, rna_id, prot_id, score in zip(organisms, rna_ids, protein_ids, scores):
        if org not in data:
            data[org] = {(rna_id, prot_id): score}
        else:
            data[org][rna_id, prot_id] = score

    common_rnas = get_common(gold_rnas, data_rnas)
    sys.stderr.write("The number of common ncRNAs is: " + str(len(common_rnas)) + "\n")
    common_prots = get_common(gold_prots, data_prots)
    sys.stderr.write("The number of common Proteins is: " + str(len(common_prots)) + "\n")

    sys.stderr.write("- Benchmarking the dataset\n")
    positive = 1
    negative = 0
    vector = []
    positives = 0
    negatives = 0
    for organism in data.keys():
        for rna, prot in data[organism].keys():
            if (rna in common_rnas) and (prot in common_prots):
                score = data[organism][(rna,prot)]
                if gold_data[organism].has_key((rna, prot)):
                    vector.append((score, positive))
                    positives += 1
                else:
                    vector.append((score, negative))
                    negatives += 1
    vector.sort(key= lambda x: x[0])

    scores, vector = zip(*vector)

    scores = moving_avg(scores, window_size)
    vector = moving_avg(vector, window_size)
    sys.stderr.write("Total positives: " + str(positives) + "," + " Total negatives: " + str(negatives) + "\n")

    f = fit_to_sigmoid(np.array(scores), np.array(vector) * SCALING_FACTOR,
                       increases=increases, fit_name=fit_name, max_value=max_value,
                       window_size=window_size, ignore_fraction=ignore_fraction)
    return map(f, np.array(original_scores))

def moving_avg(vector, window_size):
    avg_vector = []
    for i in range(len(vector) - window_size+1):
        avg_vector.append(float(sum(vector[i:window_size+i])) / float(window_size))

    return(avg_vector)

########################################################
# combine master files
########################################################

# defcombine_masterfiles(prediction_files = PREDICTION_FILES, out_file=COMBINED_PREDICTION_FILE):
def combine_masterfiles(master_files, out_file, gold_standard_file = GOLDSTANDARD_FILE, fit_name='default', window_size=25,
                        negative_evidence=False,
                        rebenchmark_everything=False, max_value=0.9, ignore_fraction=0.0, expanison = False ):

    directed_hash = {}
    evidence_hash = {}
    score_hash = collections.defaultdict(list)
    # score_hash = {}
    source_hash = collections.defaultdict(list)
    url_hash = collections.defaultdict(list)
    comment_hash = collections.defaultdict(list)
    orgn2master_idx = {} # to ensure only to penalize if the method covers the organism should one use the neg evidence option
    # parameters = (2.54666370e-01, 1.81150150e+02, 1.37511440e-01, 1.42268328e-01)

    priors = []
    prior_total = get_prior(gold_standard_file, expanison = expanison) # Should this be the prior based on the union of the master files?? Probably not much different

    master_idx_hash = collections.defaultdict(set) # for diagnosis/
    #alternative_keys_to_final_keys = {} # make sure that there is only one key per interaction
    #protein_mapper = get_string_to_alias_mapper('all', '', '', 10, 'all', True)

    for master_idx, master_file in enumerate(master_files):
        prior = get_prior(os.path.join(MASTER_DIR, master_file), expanison = expanison)
        priors.append(prior)
        print 'running %s' % master_file
        for line in open( os.path.join(MASTER_DIR, master_file), 'r'):
            if re.search('\s*#', line):
                continue
            tabs = line.rstrip('\r\n').split('\t')
            if len(tabs) == 9:
                organism, rna_id, prot_id, directed, evidence, score, source, url, comment = tabs
            else:
                organism, rna_id, prot_id, directed, evidence, score, source, url, comment = (tabs + ['', ''])[:9]
    
            score = float(score)
            if score < prior and not negative_evidence:
                continue

            # make sure that same interaction is not read twice with RNA and protein swapped -->
            # CG: potential issue with clip data e.g. DGCR8, or is that simply the cost of the reduced resolution (mRNA==protein)?
            key = (rna_id, prot_id, organism)
            '''if key_1 in alternative_keys_to_final_keys:
                key = alternative_keys_to_final_keys[key_1]
            else:
                key = key_1
                key_2 = (prot_id, rna_id, organism)
                alternative_keys_to_final_keys[key_1] = key
                alternative_keys_to_final_keys[key_2] = key
            '''
            if master_idx in master_idx_hash[key]:
                sys.stderr.write('Interaction {} appeared at least twice in {}.\n'.format(str(key), master_file))
                continue
            else:
                master_idx_hash[key].add(master_idx)

            if negative_evidence:
                if not organism in orgn2master_idx:
                    orgn2master_idx[organism] = [ False for x in master_files ]
                orgn2master_idx[organism][master_idx] = True

            directed_hash[key] = directed
            evidence_hash[key] = evidence
            score_hash[key].append((score,master_idx))
            source_hash[key].append(source)
            if url != '':
                url_hash[key].append(url)
            if comment != '':
                comment_hash[key].append(comment)

    sys.stderr.write('pre benchmarking\n')

    scores, organisms, rna_ids, prot_ids = [], [], [], []
    for idx, (key, source) in enumerate(source_hash.items()):
        if rebenchmark_everything or len(source) != 1:
            rna_id, prot_id, organism = key
            '''first_id, second_id, organism = key

            if first_id in protein_mapper[organism]:
                prot_id, rna_id = first_id, second_id
            elif second_id in protein_mapper[organism]:
                rna_id, prot_id = first_id, second_id
            else:
                sys.stderr.write('In stringrnautils.combine_masterfiles(): Neither {} nor {} were identified as protein'
                                 ' identifiers in organism {}.\n'.format(
                                 first_id, second_id, organism))
                prot_id, rna_id = first_id, second_id
            '''
            #           ___
            # (1 - p)   | |  (1 - Pi)
            # ------- = | |  --------
            # (1 - p*)  | |  (1 - Pi*)
            #
            # p = 1 - (1 - p*)^(1-N) * Poduct((1 - pi) --> this only applies when Pi*==p*
            rev_scores, master_idxes = zip(*[ [1 - s[0],s[1]] for s in score_hash[key]])
            rev_scores = list(rev_scores)
            rev_priors = [ 1 - s for idx,s in enumerate(priors) if idx in master_idxes ]
            if negative_evidence: # append p=0.0->rev_score=1.0 for not captured --> seem to cause issues with negative p scores
                rev_scores += [ 1.0 for idx in range(len(priors))
                                if (not idx in master_idxes) and orgn2master_idx[organism][idx] ]
                rev_priors += [ 1 - s for idx,s in enumerate(priors)
                                if (not idx in master_idxes) and orgn2master_idx[organism][idx] ]

            p = 1 -  (1 - prior_total) * reduce(operator.mul, rev_scores)  / reduce(operator.mul, rev_priors)
            p = p if p>0.0 else 0.0 # ensure that min score is 0.0
            scores.append(p)
            organisms.append(organism)
            rna_ids.append(rna_id)
            prot_ids.append(prot_id)
        else:
            score_hash[key] = [(score_hash[key][0][0], "NA")]

    sys.stderr.write('benchmarking\n')

    new_scores = benchmark(organisms, rna_ids, prot_ids, scores, gold_standard_file, increases=True,
                           window_size=window_size, fit_name=fit_name, max_value=max_value,
                           ignore_fraction=ignore_fraction)

    for rna_id, prot_id, organism, new_score in zip(rna_ids, prot_ids, organisms, new_scores):
        '''key_1 = (rna_id, prot_id, organism)
        key_2 = (prot_id, rna_id, organism)
        assert(alternative_keys_to_final_keys[key_1] == alternative_keys_to_final_keys[key_2])

        key = alternative_keys_to_final_keys[key_1]
        '''
        key = (rna_id, prot_id, organism)
        if key not in score_hash:
            raise ValueError('Non-existing key encountered: {}'.format(str(key)))
        else:
            score_hash[key] = [(new_score,"NA")]

    out_file = open(os.path.join(MASTER_DIR, out_file), 'w')
    for key, directed in directed_hash.items():
        first_id, second_id, organism = key

        score_val = score_hash[key]
        if len(score_val) != 1 or score_val[0][1] != 'NA':
            raise ValueError('Following key was not properly benchmarked and updated: {}\nwas mapped to: {}\n'.format(str(key), str(score_val)))


        out_data = (organism, first_id, second_id, directed_hash[key], evidence_hash[key], score_hash[key][0][0],
                    ';'.join(source_hash[key]), ';'.join(url_hash[key]),';'.join(comment_hash[key]) )

        out_file.write('\t'.join(map(str, out_data)))
        out_file.write('\n')

    out_file.close()

    # for master_file in master_files:
    #     os.unlink( os.path.join(MASTER_DIR, master_file) )


########################################################
# Misc
########################################################


def get_string_10_species():
    """

    :return: a list of strings where each string is the taxonomy identifier of a species in STRING 10
    """
    tax_list = list(species_name_to_taxonomy_id().values())
    tax_list.sort()
    return tax_list


def starbase_exp_pmids():
    """
    :return: set of ints - the PubMed IDs of publication whose experimental data sets have been integrated in StarBase
    """
    pmids = set()
    with open(STARBASE_PMIDs, 'r') as f:
        for line in f:
            if line.startswith("#"):
                continue
            line.rstrip()
            pmids.add(int(line))
    return pmids


def species_name_to_taxonomy_id():
    """

    :return: dictionary, string -> string, that maps species names to their taxonomy identifiers. Based on STRING
      species file.
    """
    taxonomy_id_to_species_name = {}
    string_species_file = os.path.join(DATA_PATH, STRING_SPECIES_100.split('/')[-1])
    if not os.path.exists(string_species_file) or DEBUGGING:
        urllib.urlretrieve(STRING_SPECIES_100, string_species_file)
    with open(string_species_file, 'r') as species_handle:
        next(species_handle)
        for line in species_handle:
            taxonomy_id, string_type, string_name, ncbi_name = line.strip().split('\t')
            taxonomy_id_to_species_name[ncbi_name.lower()] = taxonomy_id
    return taxonomy_id_to_species_name

def reduce_dict_scores( input_dict, method):
    if method == "mean":
        for key,val in input_dict.iteritems():
            input_dict[ key ] = np.array(val).mean()
    elif method == "min":
        for key,val in input_dict.iteritems():
            input_dict[ key ] = np.array(val).min()
    elif method == "max":
        for key,val in input_dict.iteritems():
            input_dict[ key ] = np.array(val).max()
    elif method == "sum":
        for key,val in input_dict.iteritems():
            input_dict[ key ] = np.array(val).sum()
    else:
        raise ValueError('In reduce_dict_scores() in stringrnautils.py: Unsupported method {}'.format(method))


def qq_correct( input_dict, ofig="None", ref_tax = "9606" ):
    """
    qq_correct: Quantile normalization of orgnisms to the human

    Arguments   Type        Description
    -----------------------------------
    input_dict  Dictionary  Two layer dictionary of type input_dict[ tax_id ][ interaction_key ]
    ofig        str         Output file name
    ref_tax     str         Taxonomy id of reference organism, default 9606
    """
    #----------------
    # Load packages
    #----------------
    from scipy.interpolate import interp1d
    from scipy import stats
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    #----------------
    # Verify arguments
    #----------------
    if not isinstance(ref_tax, str):
        raise TypeError("func qq_correct, argument human_tax must be a string")
    if not isinstance(ofig, str):
        raise TypeError("func qq_correct, argument ofig must be a string")
    if not isinstance(input_dict, dict):
        raise TypeError("func qq_correct, argument input_dict must be a dictionary")
    if not ref_tax in input_dict.keys():
        raise ValueError( "Dictionary doesn't hold the ref taxonomy %s"%ref_tax )

    #----------------
    # Define human quantiles and quantile mapping function
    #----------------
    ref_scores = np.sort( np.array( input_dict[ref_tax].values() ) )
    ref_scores_min = np.min(ref_scores)
    ref_scores_max = np.max(ref_scores)
    ref_rank_scores = stats.rankdata(ref_scores, "average")/len(ref_scores)
    ref_rank_scores_min = np.min(ref_rank_scores)
    ref_rank_scores_max = np.max(ref_rank_scores)
    qq_func = interp1d( ref_rank_scores, ref_scores, kind='linear')

    #----------------
    # perform quantile normalization
    #----------------
    pdf = None
    if not ofig=="None":
        pdf = PdfPages( os.path.join(FIG_PATH, ofig ) )
    for taxonomy in [ tax for tax in input_dict.keys() if not tax==ref_tax ]:
        keys, scores = zip(*input_dict[taxonomy].items())
        scores = np.array(scores)
        rank_scores = stats.rankdata(scores, "average")/len(scores)
        rank_scores_min = np.min(rank_scores)
        rank_scores_max = np.max(rank_scores)
        rank_scores = (rank_scores - rank_scores_min) * (ref_rank_scores_max - ref_rank_scores_min) / (rank_scores_max - rank_scores_min ) + ref_rank_scores_min
        new_scores = qq_func( rank_scores )
        new_scores[ rank_scores==ref_rank_scores_min ] = ref_scores_min # boundary issue
        new_scores[ rank_scores==ref_rank_scores_max ] = ref_scores_max # boundary issue
        input_dict[ taxonomy ] = dict([ (key,score) for key,score in zip(keys,new_scores) ])
        overall_min = np.min( (np.min(ref_scores), np.min(scores), np.min(new_scores)) )
        overall_max = np.max( (np.max(ref_scores), np.max(scores), np.max(new_scores)) )
        if not ofig=="None":
            #----------------
            # Generate histograms
            #----------------
            f, axarr = plt.subplots(2, sharex=True)
            axarr[0].hist( ref_scores, color="red",alpha=0.4,normed=True,label="Taxonomy:%s"%ref_tax,log=True,bins=100)
            axarr[0].hist( scores, color="blue",alpha=0.4,normed=True,label="Taxonomy:%s"%taxonomy,log=True,bins=100)
            axarr[0].set_xlim( (overall_min,overall_max) )
            axarr[0].set_title('No Normalization')
            axarr[0].set_ylabel("Density")
            axarr[0].legend(loc='best',frameon=False)
            #
            axarr[1].hist( ref_scores, color="red",alpha=0.4,normed=True,label="Taxonomy:%s"%ref_tax,log=True,bins=100)
            axarr[1].hist( new_scores, color="blue",alpha=0.4,normed=True,label="Taxonomy:%s"%taxonomy,log=True,bins=100)
            axarr[1].set_title('Quantile Normalization')
            axarr[1].set_xlabel("Confidence Score")
            axarr[1].set_ylabel("Density")
            axarr[1].legend(loc='best',frameon=False)
            pdf.savefig(f)
            plt.close()

    if not ofig=="None":
        pdf.close()
    return input_dict


def map_gene_2_enemble(gene2ensemble_url, gene2ensemble_file):
    ensemble_gene_dict = collections.defaultdict(dict)
    tax_idx, gene_idx, ensembl_idx = 0,3,-1

    if not os.path.exists( gene2ensemble_file ):
        urllib.urlretrieve(gene2ensemble_url, gene2ensemble_file)

    species_list = {'10090', '9606', '7227', '7955', '10116', '6239', '3702'}
    fp = gzip.open(gene2ensemble_file, 'r')
    for record in fp:
        record = record.rstrip('\r\n').split('\t')
        if record[tax_idx] in species_list and record[gene_idx] != '-' and record[ensembl_idx] != '-':
            ensemble_gene_dict[record[tax_idx]][ record[gene_idx].split('.')[0] ] = record[ensembl_idx]

    fp.close()
    return ensemble_gene_dict

def integrate_NM_dictionary(gene2ensembl):
    worm_string = get_alias_to_string_mapper('6239', '', 'NM_', 10, 'all')['6239']
    plant_string = get_alias_to_string_mapper('3702', '', '', 10, 'all')['3702']
    for key, value in worm_string.iteritems():
        gene2ensembl['6239'][key] = value
    for key, value in plant_string.iteritems():
        gene2ensembl['3702'][key] = value


def map_mirna_family(mirna_family_zip_file):
    miR_family = {}
    tax_idx, idx1, idx2 = 2,3,-1
    species = {'9606', '10090', '7955'}
    ztmp = zipfile.ZipFile(mirna_family_zip_file)
    for tmp_unzip_file in ztmp.namelist():
        fp = ztmp.open(tmp_unzip_file, 'r')
        for record in fp:
            record = record.split()
            if record[tax_idx] in species:
                miR_family[record[idx1]] = record[idx2]
        fp.close()
    return miR_family


if __name__ == '__main__':
    version = 20
    path = MIR_MAPPING_ALIASES_PATH
    make_mir_mapping_files(path, version)

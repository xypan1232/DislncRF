#!/usr/bin/env python

''' Parse object data from a .obo file

'''

from __future__ import print_function, division

import json
import pdb
from collections import defaultdict
 

class obo_object(object):
    def __init__(self, fname = "data/doid.obo"):

        self.fname = fname
        self.terms = {}
        
    def getTerm(self, stream):
        block = []
        for line in stream:
          if line.strip() == "[Term]" or line.strip() == "[Typedef]":
            break
          else:
            if line.strip() != "":
              block.append(line.strip())
        
        return block
    
    def parseTagValue(self, term):
        data = {}
        for line in term:
          tag = line.split(': ',1)[0]
          value = line.split(': ',1)[1]
          if not data.has_key(tag):
            data[tag] = []
        
          data[tag].append(value)
        
        return data

    def getDescendents(self, goid):
        recursiveArray = [goid]
        if self.terms.has_key(goid):
          children = self.terms[goid]['c']
          if len(children) > 0:
            for child in children:
              #recursiveArray.extend(self.getDescendents(child))
              recursiveArray.append(child)
        return set(recursiveArray)
  
    def getDescendent_all(self, goid):
        recursiveArray = [goid]
        if self.terms.has_key(goid):
          children = self.terms[goid]['c']
          if len(children) > 0:
            for child in children:
              recursiveArray.extend(self.getDescendent_all(child))
              #recursiveArray.append(child)
        return set(recursiveArray)
    
    def getAncestors(self, goid):
        recursiveArray = [goid]
        if self.terms.has_key(goid):
          parents = self.terms[goid]['p']
          if len(parents) > 0:
            for parent in parents:
              #recursiveArray.extend(self.getAncestors(parent))
              recursiveArray.append(parent)
        
        return set(recursiveArray)

    def getAncestor_all(self, goid):
        recursiveArray = [goid]
        if self.terms.has_key(goid):
          parents = self.terms[goid]['p']
          if len(parents) > 0:
            for parent in parents:
              recursiveArray.extend(self.getAncestor_all(parent))
              #recursiveArray.append(parent)
        
        return set(recursiveArray)
  
    def read_obo_file(self): 
        oboFile = open(self.fname,'r')
        
        #skip the file header lines
        self.getTerm(oboFile)
        
        #infinite loop to go through the obo file.
        #Breaks when the term returned is empty, indicating end of file
        while 1:
          #get the term using the two parsing functions
          term = self.parseTagValue(self.getTerm(oboFile))
          if len(term) != 0:
            termID = term['id'][0]
        
            #only add to the structure if the term has a is_a tag
            #the is_a value contain GOID and term definition
            #we only want the GOID
            if term.has_key('is_a'):
              termParents = [p.split()[0] for p in term['is_a']]
        
              if not self.terms.has_key(termID):
                #each goid will have two arrays of parents and children
                self.terms[termID] = {'p':[],'c':[]}
        
              #append parents of the current term
              self.terms[termID]['p'] = termParents
        
              #for every parent term, add this current term as children
              for termParent in termParents:
                if not self.terms.has_key(termParent):
                  self.terms[termParent] = {'p':[],'c':[]}
                self.terms[termParent]['c'].append(termID)
          else:
            break

if __name__ == '__main__':
    oboparser = obo_object()
    oboparser.read_obo_file()
    pdb.set_trace()
    resu=oboparser.getAncestors('DOID:3459')
    
    

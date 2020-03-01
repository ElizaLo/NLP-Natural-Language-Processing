#!/usr/bin/env python

"""
Penn Discourse Treebank 2.0 reader, requiring the PDTB file format
used for the materials at http://compprag.christopherpotts.net/
"""

__author__ = "Christopher Potts"
__copyright__ = "Copyright 2011, Christopher Potts"
__credits__ = []
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__version__ = "1.0"
__maintainer__ = "Christopher Potts"
__email__ = "See the author's website"

# Converted to Python 3 by Tatjana Scheffler, 2019
		
######################################################################

import os
import sys
import csv
import re
from collections import defaultdict
from nltk.tree import Tree
from nltk.stem import WordNetLemmatizer

GRAPHVIZ_TEMPLATE_FILENAME = 'pdtb-template.dot'

######################################################################

class CorpusReader:
    def __init__(self, src_filename):
        self.src_filename = src_filename
   
    def iter_data(self, display_progress=True):
        row_iterator = csv.reader(open(self.src_filename,'rU'))
        next(row_iterator) # Skip past the header.
        i = 1
        for row in row_iterator:
            if display_progress:
                sys.stderr.write("\r") ; sys.stderr.write("row %s" % i) ; sys.stderr.flush()
                i += 1
            yield Datum(row)
        if display_progress: sys.stderr.write("\n")

######################################################################

class Datum:

    header = [
    # Corpus
    'Relation', 'Section', 'FileNumber',
    ##### Connective.
    'Connective_SpanList', 'Connective_GornList', 'Connective_Trees', 'Connective_RawText', 'Connective_StringPosition',
    'SentenceNumber', 'ConnHead', 'Conn1', 'Conn2', 'ConnHeadSemClass1', 'ConnHeadSemClass2', 'Conn2SemClass1', 'Conn2SemClass2',
    # Connective attribution.
    'Attribution_Source', 'Attribution_Type', 'Attribution_Polarity', 'Attribution_Determinacy',
    'Attribution_SpanList', 'Attribution_GornList', 'Attribution_Trees', 'Attribution_RawText',
    ##### Arg1
    'Arg1_SpanList', 'Arg1_GornList', 'Arg1_Trees', 'Arg1_RawText',
    # Arg1 Attribution.
    'Arg1_Attribution_Source', 'Arg1_Attribution_Type', 'Arg1_Attribution_Polarity', 'Arg1_Attribution_Determinacy',
    'Arg1_Attribution_SpanList', 'Arg1_Attribution_GornList', 'Arg1_Attribution_Trees', 'Arg1_Attribution_RawText',
    ##### Arg2.
    'Arg2_SpanList', 'Arg2_GornList', 'Arg2_Trees', 'Arg2_RawText',
    # Arg2 Attribution.
    'Arg2_Attribution_Source', 'Arg2_Attribution_Type', 'Arg2_Attribution_Polarity', 'Arg2_Attribution_Determinacy',
    'Arg2_Attribution_SpanList', 'Arg2_Attribution_GornList', 'Arg2_Attribution_Trees', 'Arg2_Attribution_RawText',
    ##### Sup1.
    'Sup1_SpanList', 'Sup1_GornList', 'Sup1_Trees', 'Sup1_RawText',
    ##### Sup2
    'Sup2_SpanList', 'Sup2_GornList', 'Sup2_Trees', 'Sup2_RawText',
    ##### Full raw text.
    'FullRawText'
    ]
    
    def __init__(self, row):
        """        
        Where row is a list, it is processed directly.
        Where row is a string, it is parsed into a list using a csv.reader.
        The attribute names are given in the class variable Datum.header.
        """
        if row.__class__.__name__ in ('str', 'unicode'):
            row = list(csv.reader([row.strip()]))[0]        
        # Set the attributes.
        for i in range(len(row)):
            att_name = Datum.header[i]
            row_value = row[i]
            # Span lists.
            if re.search(r"SpanList", att_name):
                row_value = self.__process_span_list(row_value)
            # Gorn lists.
            elif re.search(r"GornList", att_name):
                row_value = self.__process_gorn_list(row_value)
            # Integer-valued.
            elif att_name in ('Connective_StringPosition', 'SentenceNumber'):
                if row_value:
                    row_value = int(row_value)
                else:
                    row_value = None
            # Trees.
            elif re.search(r"Trees", att_name):
                row_value = self.__process_trees(row_value)            
                # The rest are strings and so don't require special handling.
            elif not row_value:
                row_value = None
            setattr(self, att_name, row_value)

    ######################################################################
    # SEMANTIC VALUES

    def primary_semclass1(self):
        """
        ConnHeadSemClass1 has fields separated by dots. This function
        returns the first (as a str), which is always one of
        
        Comparison, Contingency, Expansion, Temporal
        """
        return self.semclass1_values()[0]

    def secondary_semclass1(self):
        """
        ConnHeadSemClass1 has fields separated by dots. This function
        returns the second (as a str) if there is one, else None.

        Values (except None):

        Alternative, Asynchronous, Cause, Concession, Condition,
        Conjunction, Contrast, Exception, Instantiation, List,
        Pragmatic cause, Pragmatic concession, Pragmatic condition,
        Pragmatic contrast, Restatement, Synchrony
        """
        vals = self.semclass1_values()
        if len(vals) >= 2:
            return vals[1]
        else:            
            return None

    def tertiary_semclass1(self):
        """
        ConnHeadSemClass1 has fields separated by dots. This function
        returns the third (as a str) if there is one, else None.

        Values (except None):

        Chosen alternative, Conjunctive, Contra-expectation,
        Disjunctive, Equivalence, Expectation, Factual past, Factual
        present, General, Generalization, Hypothetical, Implicit
        assertion, Justification, Juxtaposition, NONE, Opposition,
        Precedence, Reason, Relevance, Result,Specification,
        Succession, Unreal past, Unreal present
        """
        vals = self.semclass1_values()
        if len(vals) >= 3:
            return vals[2]
        else:            
            return None    

    def semclass1_values(self):
        if self.ConnHeadSemClass1:
            return self.ConnHeadSemClass1.split(".")
        else:
            return [None]

    ######################################################################
    # TOKENIZING AND POS-TAGGING WITH THE OPTION TO CONVERT
    # CONVERTABLE TAGS TO WORDNET STYLE

    def arg1_words(self, lemmatize=False):
        """
        Returns the list of words associated with Arg1. lemmatize=True
        uses nltk.stem.WordNetStemmer() on the list.
        """
        return self.__words(self.arg1_pos, lemmatize=lemmatize)

    def arg2_words(self, lemmatize=False):
        """
        Returns the list of words associated with Arg2. lemmatize=True
        uses nltk.stem.WordNetStemmer() on the list.
        """
        return self.__words(self.arg2_pos, lemmatize=lemmatize)

    def arg1_attribution_words(self, lemmatize=False):
        """
        Returns the list of words associated with Arg1's attrbution
        (if any). lemmatize=True uses nltk.stem.WordNetStemmer() on
        the list.
        """
        return self.__words(self.arg1_attribution_pos, lemmatize=lemmatize)

    def arg2_attribution_words(self, lemmatize=False):
        """
        Returns the list of words associated with Arg2's attrbution
        (if any). lemmatize=True uses nltk.stem.WordNetStemmer() on
        the list.
        """
        return self.__words(self.arg2_attribution_pos, lemmatize=lemmatize)

    def connective_words(self, lemmatize=False):
        """
        Returns the list of words associated with an Explicit or
        AltLex connective (else it always returns []). lemmatize=True
        uses nltk.stem.WordNetStemmer() on the list.
        """
        return self.__words(self.connective_pos, lemmatize=lemmatize)

    def sup1_words(self, lemmatize=False):
        return self.__words(self.sup1_pos, lemmatize=lemmatize)

    def sup2_words(self, lemmatize=False):
        """
        Returns the list of words associated with Sup1 (if
        any). lemmatize=True uses nltk.stem.WordNetStemmer() on the
        list.
        """
        return self.__words(self.sup2_pos, lemmatize=lemmatize)

    def __words(self, method, lemmatize=False):
        """
        Internal method used by the X_words functions to get at their
        (possibly stemmed) words.
        """
        lemmas = method(lemmatize=lemmatize)
        return [x[0] for x in lemmas]
        
    def arg1_pos(self, wn_format=False, lemmatize=False):
        """
        Returns the list of (word, pos) pairs associated with
        Arg1. lemmatize=True uses nltk.stem.WordNetStemmer() on the list.
        """
        return self.arg_pos(1, wn_format=wn_format, lemmatize=lemmatize)

    def arg2_pos(self,  wn_format=False, lemmatize=False):
        """
        Returns the list of (word, pos) pairs associated with
        Arg2. lemmatize=True uses nltk.stem.WordNetStemmer() on the list.
        """
        return self.arg_pos(2,  wn_format=wn_format, lemmatize=lemmatize)

    def arg_pos(self, index, wn_format=False, lemmatize=False):
        """
        Returns the list of (word, pos) pairs associated with ArgN,
        where N = index (1 or 2). lemmatize=True uses
        nltk.stem.WordNetStemmer() on the list. wn_format=True merely
        converts to WordNet tags where possible, without stemming.
        """
        return self.__pos("Arg%s" % index, wn_format=wn_format, lemmatize=lemmatize)

    def arg1_attribution_pos(self, wn_format=False, lemmatize=False):
        """
        Returns the list of (word, pos) pairs associated with
        Arg1. lemmatize=True uses nltk.stem.WordNetStemmer() on the
        list. wn_format=True merely converts to WordNet tags where
        possible, without stemming.
        """
        return self.arg_attribution_pos(1, wn_format=wn_format, lemmatize=lemmatize)

    def arg2_attribution_pos(self, wn_format=False, lemmatize=False):
        """
        Returns the list of (word, pos) pairs associated with
        Arg2. lemmatize=True uses nltk.stem.WordNetStemmer() on the
        list. wn_format=True merely converts to WordNet tags where
        possible, without stemming.
        """
        return self.arg_attribution_pos(2, wn_format=wn_format, lemmatize=lemmatize)

    def arg_attribution_pos(self, index, wn_format=False, lemmatize=False):
        """
        Returns the list of (word, pos) pairs associated with ArgN's
        attribution, where N = index (1 or 2). lemmatize=True uses
        nltk.stem.WordNetStemmer() on the list. wn_format=True merely
        converts to WordNet tags where possibl, without stemming.
        """
        return self.__pos("Arg%s_Attribution" % index, wn_format=wn_format, lemmatize=lemmatize)

    def connective_pos(self, wn_format=False, lemmatize=False):
        """
        Returns the list of (word, pos) pairs associated with an
        Explicit or AltLex connective (else returns []).
        lemmatize=True uses nltk.stem.WordNetStemmer() on the
        list. wn_format=True merely converts to WordNet tags where
        possibl, without stemming.
        """
        return self.__pos("Connective", wn_format=wn_format, lemmatize=lemmatize)

    def sup1_pos(self, wn_format=False, lemmatize=False):
        """
        Returns the list of (word, pos) pairs associated with Sup1 (if
        any). lemmatize=True uses nltk.stem.WordNetStemmer() on the
        list. wn_format=True merely converts to WordNet tags where
        possibl, without stemming.
        """
        return self.sup_pos(1, wn_format=wn_format, lemmatize=lemmatize)

    def sup2_pos(self, wn_format=False, lemmatize=False):
        """
        Returns the list of (word, pos) pairs associated with Sup2 (if
        any). lemmatize=True uses nltk.stem.WordNetStemmer() on the
        list. wn_format=True merely converts to WordNet tags where
        possibl, without stemming.
        """
        return self.sup_pos(2, wn_format=wn_format, lemmatize=lemmatize)

    def sup_pos(self, index, wn_format=False, lemmatize=False):
        """
        Returns the list of (word, pos) pairs (if any) associated with
        SupN where N = index (1 or 2). lemmatize=True uses
        nltk.stem.WordNetStemmer() on the list. wn_format=True merely
        converts to WordNet tags where possibl, without stemming.
        """
        return self.__pos("Sup%s" % index, wn_format=wn_format, lemmatize=lemmatize)

    def __pos(self, typ, wn_format=False, lemmatize=False):
        """
        Internal method used to get the POS version, potentially lemmatized, associated with
        typ strings (Arg1, Sup2, etc.)
        """
        results = []
        trees = eval("self.%s_Trees" % typ)
        for tree in trees:
            results += tree.pos()
        # Lemmatizing implies converting to WordNet tags.
        if lemmatize:
            results = list(map(self.__treebank2wn_pos, results))
            results = list(map(self.__lemmatize, results))
        # This is tag conversion without lemmatizing.
        elif wn_format:
            results = list(map(self.__treebank2wn_pos, results))
        return results

    def __treebank2wn_pos(self, lemma):
        """
        Internal method used for converting a lemma's tag to WordNet format where possible.
        """
        string, tag = lemma
        tag = tag.lower()
        if tag.startswith('v'):
            tag = 'v'
        elif tag.startswith('n'):
            tag = 'n'
        elif tag.startswith('j'):
            tag = 'a'
        elif tag.startswith('rb'):
            tag = 'r'
        return (string, tag)

    def __lemmatize(self, lemma):
        """
        Internal method used for applying the nltk.stem.WordNetStemmer() to the (word, pos) pair lemma.
        """
        string, tag = lemma
        if tag in ('a', 'n', 'r', 'v'):        
            wnl = WordNetLemmatizer()
            string = wnl.lemmatize(string, tag)
        return (string, tag)

    ######################################################################    
    # POSITIONING.

    def relative_arg_order(self):
        """
        1S ... 1F ... 2s ... 2f -> arg1_precedes_arg2
        1S ... 2s ... 2f ... 1F -> arg1_contains_arg2
        1S ... 2s ... 1F ... 2f -> arg1_precedes_and_overlaps_but_does_not_contain_arg2
        
        2S ... 2F ... 1S ... 1F -> arg2_precedes_arg1
        2S ... 1S ... 1F ... 2F -> arg2_contains_arg1
        2S ... 1S ... 2F ... 2F -> arg2_precedes_and_overlaps_but_does_not_contain_arg1
        """
        arg1_indices =  [i for span in self.Arg1_SpanList for i in span]
        arg1_start = min(arg1_indices)
        arg1_finish = max(arg1_indices)
        arg2_indices = [i for span in self.Arg2_SpanList for i in span]      
        arg2_start = min(arg2_indices)
        arg2_finish = max(arg2_indices)
        # Arg1 preceding:
        if arg1_finish < arg2_start:
            return 'arg1_precedes_arg2'
        if arg1_start < arg2_start and arg2_finish < arg1_finish:
            return 'arg1_contains_arg2'
        if arg1_start < arg2_start and arg2_start < arg1_finish and arg1_finish < arg2_finish:
            return 'arg1_precedes_and_overlaps_but_does_not_contain_arg2'
        # Arg2 preceding:
        if arg2_finish < arg1_start:
            return 'arg2_precedes_arg1'
        if arg2_start < arg1_start and arg1_finish < arg2_finish:
            return 'arg2_contains_arg1'
        if arg2_start < arg2_start and arg1_start < arg2_finish and arg2_finish < arg1_finish:
            return 'arg2_precedes_and_overlaps_but_does_not_contain_arg1'
        raise Exception("No relation could be determined for the two arguments!\n%s" % self.FullRawText)

    def arg1_precedes_arg2(self):
        """
        Returns True if the whole of Arg1 precedes the whole of Arg2,
        else False. (So the False includes both where Arg2 precedes
        Arg1 and where they are in some kind of overlap relationship).

        1S ... 1F ... 2s ... 2f
        """
        if self.relative_arg_order() == 'arg1_precedes_arg2':
            return True
        else:
            return False

    def arg2_precedes_arg1(self):
        """
        Returns True if the whole of Arg2 precedes the whole of Arg1,
        else False. (So the False includes both where Arg1 precedes
        Arg2 and where they are in some kind of overlap relationship).

        2S ... 2F ... 1S ... 1F
        """
        if self.relative_arg_order() == 'arg2_precedes_arg1':
            return True
        else:
            return False

    def arg1_contains_arg2(self):
        """
        Returns True if Arg1 contains Arg2 completely, else False.

        1S ... 2s ... 2f ... 1F
        """
        if self.relative_arg_order() == 'arg1_contains_arg2':
            return True
        else:
            return False

    def arg2_contains_arg1(self):
        """
        Returns True if Arg1 contains Arg2 completely, else False.

        2S ... 1S ... 1F ... 2F 
        """
        if self.relative_arg_order() == 'arg2_contains_arg1':
            return True
        else:
            return False

    def arg1_precedes_and_overlaps_but_does_not_contain_arg2(self):
        """
        Arg1 begins before Arg2, but Arg1 also ends before Arg2:
        
        1S ... 2s ... 1F ... 2f

        It seems that this is not attested in the data.
        """
        if self.relative_arg_order() == 'arg1_precedes_and_overlaps_but_does_not_contain_arg2':
            return True
        else:
            return False 

    def arg2_precedes_and_overlaps_but_does_not_contain_arg1(self):
        """
        Arg2 begins before Arg1, but Arg2 also ends before Arg1:

        2S ... 1S ... 2F ... 2F
        
        It seems that this is not attested in the data.
        """
        if self.relative_arg_order() == 'arg2_precedes_and_overlaps_but_does_not_contain_arg1':
            return True
        else:
            return False        

    ######################################################################
    # NORMALIZATION.

    def conn_str(self, distinguish_implicit=True):
        """
        Provides a method for looking at the intuitive main element of
        a connective despite the variation across relation types
        in what the main element is.

        Optional argument:

        distinguish_implicit -- prefixes Implicit= to the relation
        where appropriate (default: True)

        Value: a str
        """
        rel = self.Relation
        if rel == 'Explicit':
            return self.ConnHead
        elif rel == 'AltLex':
            return self.Connective_RawText
        elif rel == 'Implicit':
            prefix = ""
            if distinguish_implicit:
                prefix = "Implicit="            
            return prefix + self.Conn1
        else:
            return None

    def final_arg1_attribution_source(self):
        """Follows inhereted attribution values for Arg1."""
        return self.final_arg_attribution_source(1)

    def final_arg2_attribution_source(self):
        """Follows inhereted attribution values for Arg2."""
        return self.final_arg_attribution_source(2)

    def final_arg_attribution_source(self, index):
        """
        Where the attribution on an argument is Inh (inherited),
        supply the inherited value (from the connective).

        Argument:
        index -- 1 or 2 depending on the argument sought

        Value: a str
        """
        if index not in (1,2):
            raise ArgumentError('index must be int 1 or int 2; was %s (type %s).\n' % (index, index.__class__.__.name__))
        src = eval("self.Arg%s_Attribution_Source" % index)
        if src == "Inh":
            src = self.Attribution_Source
        return src

    ######################################################################
    # Vizualization.

    def to_graphviz(self, include_ptb=True):
        """
        Uses the Graphviz template to create Graphviz code for this Datum
        """
        try:
            template = open(GRAPHVIZ_TEMPLATE_FILENAME).read()
        except:
            raise Exception("Can't find the Graphviz template file %s" % GRAPHVIZ_TEMPLATE_FILENAME)
        # Title.
        new_line_removal = re.compile(r"\n\s*", re.M)
        cleaned_text = new_line_removal.sub(" ", self.FullRawText)
        title = '"%s\\n%s\\nSource: %s/wsj_00%s"' % (self.Relation, cleaned_text, self.Section, self.FileNumber)
        template = template.replace("$TITLE", title)        
        # Attributes.
        attributes = dir(self)
        for att in attributes:
            # Get the value for this attribute.
            val = getattr(self, att)
            # This is what we'll use for the label.
            val_str = '""'
            # Trees are handled specially.
            if re.search(r"_Trees", att):
                if include_ptb:
                    val_str = self.__format_graphviz_trees(val)
                else:
                    val_str = '"(not pictured)"'
            elif val == "Inh":
                val_str = '"%s / %s"' % (val, self.Attribution_Source)
            else:                
                val_str = '"%s"' % val            
            # Identify the variable in the template.
            att_re = re.compile(r'"\$' + att.upper() + r'"', re.MULTILINE)            
            # Substitution string.            
            template = att_re.sub(val_str, template)
        # Remove attributes that have None as their value.
        for att in attributes:
            val = getattr(self, att)
            if not val:
                template = re.sub(r".*\s%s\s.*" % att, "", template)
        # Remove blank lines to keep the file neat.
        blank_line_re = re.compile("^\s*$", re.M)
        template = blank_line_re.sub("", template)
        # Return the graphviz code, as a multiline string.
        return template

    def __format_graphviz_trees(self, trees):
        """Internal method for formatting trees for Graphviz node labels."""
        if not trees:
            return '""'
        s = "<<table border='0'>"
        for tree in trees:
            s += "<tr><td>%s</td></tr>" % re.sub(r"\n\s*", " ", str(tree))
        s += "</table>>"
        return s

    def __str__(self):
        """Just returns the full text of the string."""
        return self.FullRawText
                
    ######################################################################
    # INTERNAL HELPER METHODS.

    def __process_span_list(self, s):
        """
        Argument
        s (str) -- of the form i..j;k..m;

        Value
        A list of pairs (list) of integers.
        """
        if not s:
            return []
        parts = re.split(r"\s*;\s*", s)
        seqs = list(map((lambda x : list(map(int, re.split(r"\s*\.\.\s*", x)))), parts))
        return seqs

    def __process_gorn_list(self, s):
        """
        Argument
        s (str) -- of the form i,j,...;k,l,...

        Value
        A list of lists (any length) of integers.
        """
        if not s:
            return []
        parts = re.split(r"\s*;\s*", s)
        seqs = list(map((lambda x : list(map(int, re.split(r"\s*,\s*", x)))), parts))
        return seqs

    def __process_trees(self, s):
        """
        Input
        a string representing Penn parsetrees, delimited by |||

        Value:
        A list of NLTK Tree objects.
        """
        if not s:
            return []
        tree_strs = s.split("|||")
        return list(map(Tree.fromstring, tree_strs)) ## important fix TS

######################################################################
            
if __name__ == '__main__':
    """
    Will try to map a string representation of the row supplied to its
    Graphviz code, along quotation marks and the like make this hard
    to achieve at the command-line.
    """
    s = sys.argv[1]
    d = Datum(s)
    print(d.to_graphviz())
        
    
        
        

       

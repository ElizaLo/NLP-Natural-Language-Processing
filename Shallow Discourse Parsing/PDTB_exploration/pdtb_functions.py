#!/usr/bin/env python

"""
Miscellaneous functions for exploring the Penn Discourse Treebank via
the classes in pdtb.py
"""

__author__ = "Christopher Potts"
__copyright__ = "Copyright 2011, Christopher Potts"
__credits__ = []
__license__ = "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License: http://creativecommons.org/licenses/by-nc-sa/3.0/"
__version__ = "1.0"
__maintainer__ = "Christopher Potts"
__email__ = "See the author's website"

# converted to Python 3 by Tatjana Scheffler, 2019
		
######################################################################

import re
import csv
import pickle
import numpy
from random import shuffle
from collections import defaultdict
from operator import itemgetter
from pdtb import CorpusReader, Datum

######################################################################
 
def relation_count():
    """Calculate and display the distribution of relations."""
    pdtb = CorpusReader('pdtb2.csv')
    # Create a count dictionary of relations:
    d = defaultdict(int)
    for datum in pdtb.iter_data():
        d[datum.Relation] += 1
    # Print the results to standard output:
    for key, val in d.items():
        print(key, val)
 
######################################################################

def count_semantic_classes():
    """Count ConnHeadSemClass1 values."""
    pdtb = CorpusReader('pdtb2.csv')
    d = defaultdict(int)
    for datum in pdtb.iter_data():
        sc = datum.ConnHeadSemClass1
        # Filter None values (should be just EntRel/NonRel data):
        if sc:
            d[sc] += 1
    return d

def count_semantic_classes_to_csv(output_filename):
    """Write the results of  count_semantic_classes() to a CSV file."""
    # Create the CSV writer:
    csvwriter = csv.writer(file(output_filename, 'w'))
    # Add the header row:
    csvwriter.writerow(['ConnHeadSemClass1', 'Count'])
    # Get the counts:
    d = count_semantic_classes()
    # Sort by name so that we can perhaps see trends in the
    # super-categories:
    for sem, count in sorted(d.items()):
        csvwriter.writerow([sem, count])
 
# count_semantic_classes_to_csv('ConnHeadSemClass1.csv')

######################################################################

def connective_distribution():
    """Counts of connectives by relation type."""
    pdtb = CorpusReader('pdtb2.csv')
    d = defaultdict(lambda : defaultdict(int))
    for datum in pdtb.iter_data():
        cs = datum.conn_str(distinguish_implicit=False)
        # Filter None values (should be just EntRel/NoRel data):
        if cs:
            # Downcase for further collapsing, and add 1:
            d[datum.Relation][cs.lower()] += 1
    return d

def connective_distribution2wordle(d):
    """
    Map the dictionary returned by connective_distribution() to a
    Wordle format. The return value is a string. Its sublists it
    returned can be pasted in at http://www.wordle.net/advanced.
    """
    s = ''
    # Print lists of words with the relation type as the header:
    for rel, counts in list(d.items()):
        s += '======================================================================\n'
        s += rel + '\n'
        s += '======================================================================\n'
        # Map the counts dict to a list of pairs via items() and sort on
        # the second member (index 1) of those pairs, largest to smallest:
        sorted_counts = sorted(list(counts.items()), key=itemgetter(1), reverse=True)
        # Print the result in Wordle format:
        for conn, c in sorted_counts:
            # Spacing is hard to interpret in Wordle. This should help:
            conn = conn.replace(' ', '_')
            # Append to the growing string:
            s += '%s:%s\n' % (conn, c)
    return s

######################################################################

def attribution_counts():
    """Create a count dictionary of non-null attribution values."""
    pdtb = CorpusReader('pdtb2.csv')
    d = defaultdict(int)
    for datum in pdtb.iter_data():
        src = datum.Attribution_Source
        if src:
            d[src] += 1
    return d

def print_attribution_texts():
    """Inspect the strings characterizing attribution values."""
    pdtb = CorpusReader('pdtb2.csv')
    for datum in pdtb.iter_data(display_progress=False):
        txt = datum.Attribution_RawText
        if txt:
            print(txt)

######################################################################

def adjacency_check(datum):
    """Return True if datum is of the form Arg1 (connective) Arg2, else False"""    
    if not datum.arg1_precedes_arg2():
        return False
    arg1_finish = max([x for span in datum.Arg1_SpanList for x in span])
    arg2_start = min([x for span in datum.Arg2_SpanList for x in span])    
    if datum.Relation == 'Implicit':
        if (arg2_start - arg1_finish) <= 3:
            return True
        else:
            return False
    else:
        conn_indices = [x for span in datum.Connective_SpanList for x in span]
        conn_start = min(conn_indices)
        conn_finish = max(conn_indices)
        if (conn_start - arg1_finish) <= 3 and (arg2_start - conn_finish) <= 3:
            return True
        else:
            return False        

def connective_initial(sem_re, output_filename):
    """
    Pull out examples of Explicit or Implicit relations in which

    (i) Arg1 immediately precedes Arg2, with only the connective intervening in the case of Explicit.
    (ii) There is no supplementary text on either argument.
    (iii) ConnHeadSemClass1 matches the user-supplied regex sem_re    

    The results go into a CSV file named output_filename.
    """
    keepers = {} # Stores the items that pass muster.
    pdtb = CorpusReader('pdtb2.csv')
    for datum in pdtb.iter_data(display_progress=False):
        # Restrict to examples that are either Implicit or Explicit and have no supplementary text:
        rel = datum.Relation
        if rel in ('Implicit', 'Explicit') and not datum.Sup1_RawText and not datum.Sup2_RawText:
            # Further restrict to the class of semantic relations captured by sem_re:            
            if sem_re.search(datum.ConnHeadSemClass1):                
                # Make sure that Arg1, the connective, and Arg2 are all adjacent:
                if adjacency_check(datum):
                    # Stick to simple connectives: for Explicit, the connective and its head are the same;
                    # for Implicit, there is no secondary connective.
                    if (rel == 'Explicit' and datum.ConnHead == datum.Connective_RawText) or \
                       (rel == 'Implicit' and not datum.Conn2):
                        itemId = "%s/%s" % (datum.Section, datum.FileNumber)
                        print(itemId)
                        conn = datum.conn_str(distinguish_implicit=False) # We needn't flag them, since column 2 does that.
                        # Store in a dict with file number keys to avoid taking two sentences from the same file:
                        keepers[itemId] = [itemId, rel, datum.ConnHeadSemClass1, datum.Arg1_RawText, conn, datum.Arg2_RawText]
    # Store the results in a CSV file:
    csvwriter = csv.writer(file(output_filename, 'w'))
    csvwriter.writerow(['ItemId', 'Relation', 'ConnHeadSemClass1', 'Arg1', 'Connective', 'Arg2'])
    csvwriter.writerows(list(keepers.values()))
    print("CSV created.")
    
# connective_initial(re.compile(r'Expansion'), 'pdtb-continuation-data-expansion.csv')    

######################################################################

def semantic_classes_in_implicit_relations():
    """Count the primary semantic classes for connectives limted to Implicit relations."""
    d = defaultdict(int)
    pdtb = CorpusReader('pdtb2.csv')
    for datum in pdtb.iter_data(display_progress=True):
        if datum.Relation == 'Implicit':
            d[datum.primary_semclass1()] += 1
    # Print, sorted by values, largest first:
    for key, val in sorted(list(d.items()), key=itemgetter(1), reverse=True):
        print(key, val)

# semantic_classes_in_implicit_relations()

######################################################################

def word_pair_frequencies(output_filename):
    """
    Gather count data on word pairs where the first word is drawn from
    Arg1 and the second from Arg2. The results are storied in the
    pickle file output_filename.
    """
    d = defaultdict(int)
    pdtb = CorpusReader('pdtb2.csv')
    for datum in pdtb.iter_data(display_progress=True):
        if datum.Relation == 'Implicit':
            # Gather the word-pair features for inclusion in d.
            # See the Datum methods arg1_words() and arg2_words.
            pass
            
    # Finally, pickle the results.
    pickle.dump(d, file(output_filename, 'w'))

######################################################################

def random_Implicit_subset(sample_size=30):
    """
    Creates a CSV file containing randomly selected Implicit examples
    from each of the primary semantic classes. sample_size determines
    the size of the sample from each class (default: 30). The output
    is a file called pdtb-random-Implicit-subset.csv with columns named for
    the attributes/methods that determined the values.
    """
    csvwriter = csv.writer(open('pdtb-random-Implicit-subset.csv', 'w'))
    csvwriter.writerow(['Arg1_RawText', 'conn_str', 'Arg2_RawText', 'primary_semclass1'])
    d = defaultdict(list)
    pdtb = CorpusReader('pdtb2.csv')
    for datum in pdtb.iter_data(display_progress=True):
        if datum.Relation == 'Implicit' and not datum.Sup1_RawText and not datum.Sup2_RawText:
            d[datum.primary_semclass1()].append(datum)
    sample_size = 30
    for cls, data, in list(d.items()):
        shuffle(data)
        for datum in data[: sample_size]:
            row = [datum.Arg1_RawText, datum.conn_str(), datum.Arg2_RawText, cls]
            csvwriter.writerow(row)

######################################################################

def distribution_of_relative_arg_order():
    d = defaultdict(int)
    pdtb = CorpusReader('pdtb2.csv')
    for datum in pdtb.iter_data(display_progress=True):
        d[datum.relative_arg_order()] += 1
    for order, count in sorted(list(d.items()), key=itemgetter(1), reverse=True):
        print(order, count)
    
######################################################################

def contingencies(row_function, column_function):
    """
    Calculates observed/expected values for the count matrix
    determined by row_function and column_function, both of which
    should be functions from Datum instances into values.  Output
    values of None are ignored and can thus be used as a filter.
    """
    d = defaultdict(int)
    pdtb = CorpusReader('pdtb2.csv')
    for datum in pdtb.iter_data(display_progress=True):
        row_val = row_function(datum)
        col_val = column_function(datum)
        if row_val and col_val:
            d[(row_val, col_val)] += 1
    # Convert to matrix format:
    row_classes = sorted(set([x[0] for x in list(d.keys())]))
    col_classes = sorted(set([x[1] for x in list(d.keys())]))
    observed = numpy.zeros((len(row_classes), len(col_classes)))
    for i, row in enumerate(row_classes):
        for j, col in enumerate(col_classes):
            observed[i, j] = d[(row, col)]
    # Get expectations:
    expected = numpy.zeros((len(row_classes), len(col_classes)))
    for i in range(len(row_classes)):
        for j in range(len(col_classes)):
            expected[i, j] = (numpy.sum(observed[i, : ]) * numpy.sum(observed[ :, j])) / numpy.sum(observed)
    # Rank by O/E:
    oe = {}
    for i, row in enumerate(row_classes):
        for j, col in enumerate(col_classes):
            oe[(row, col)] = observed[i, j] / expected[i, j]
    # Printing:
    print("======================================================================")
    print("Observed")
    print_matrix(observed, row_classes, col_classes)
    print("--------------------------------------------------")
    print("Expected")
    print_matrix(expected, row_classes, col_classes)
    print("--------------------------------------------------")
    print("O/E")
    for row, col in sorted(list(oe.items()), key=itemgetter(1), reverse=True):
        print(row, col)
    print("======================================================================")

def print_matrix(m, rownames, colnames):
    """Pretty-print a 2d numpy array with row and column names."""
    # Max column width:
    col_width = max([len(x) for x in rownames + colnames]) + 4
    # Row-formatter:
    def fmt_row(a):
        return "".join(map((lambda x : str(x).rjust(col_width)), a))
    # Printing:
    print(fmt_row([''] + colnames))            
    for i, rowname in enumerate(rownames):
        row = [rowname]
        for j in range(len(colnames)):
            row.append(round(m[i, j], 2))
        print(fmt_row(row))
    
    
        
    
        
    

"""
USE: python <PROGNAME> (options) 
OPTIONS:
    -h : print this help message and exit
    -d FILE : use FILE as data to create a new lexicon file
    -l FILE : create OR read lexicon file FILE
    -t FILE : apply lexicon to test data in FILE
"""
################################################################

import sys, re, getopt
from collections import OrderedDict
import operator

################################################################
# Command line options handling, and help

opts, args = getopt.getopt(sys.argv[1:],'hd:l:t:')
opts = dict(opts)
'''
def printHelp():
    help = __doc__.replace('<PROGNAME>',sys.argv[0],1)
    print('-' * 60, help, '-' * 60, file=sys.stderr)
    sys.exit()
    
if '-h' in opts:
    printHelp()

if len(args) > 0:
    print("\n** ERROR: no arg files - only options! **", file=sys.stderr)
    printHelp()

if '-l' not in opts:
    print("\n** ERROR: must specify lexicon file name (opt: -l) **", file=sys.stderr)
    printHelp()
'''
################################################################

def analyse_tagged_text():
    term_dict = sort_terms()
    #print_tags(term_dict)
    #check_ambig(term_dict)
    #total_tags = count_tags(term_dict)
    naive_dict = naive_tags(term_dict)
    

def sort_terms():
    term_dict = {}
    word_pos = re.compile("'\w/+\w+|\w+/\w+\w+")
    file = open('POSTAG_DATA/training_data.txt')
    for line in file:
        for word in word_pos.findall(line):
            split = (word.split('/'))
            term = split[0]
            pos = split[1]
            if term not in term_dict:
                term_dict[term] = {pos: 1}
            else:
                if pos not in term_dict[term]:
                    term_dict[term][pos] = 1
                else:
                    term_dict[term][pos] = (term_dict[term][pos]+1)
    return(term_dict)


def print_tags(term_dict):
    list_of_pos = {}
    for key in term_dict:
        for pos in term_dict[key]:
            if pos not in list_of_pos:
                list_of_pos[pos] = 1
            else:
                list_of_pos[pos] = list_of_pos[pos]+1
    list_of_pos = OrderedDict(sorted(list_of_pos.items(), key=itemgetter(1), reverse=True))

    for pos in list_of_pos:
        print(pos + ': ' + str(list_of_pos[pos]))

def count_tags(term_dict):
    tag_counter = 0
    for key in term_dict:
        for pos in term_dict[key]:
            tag_counter += term_dict[key][pos]
    return tag_counter

def check_ambig(term_dict):
    counter = 0
    for key in term_dict:
        if len(term_dict[key]) > 1:
            counter+=1
    terms = len(term_dict)
    print(str(terms) + ' Total terms')
    print(str(counter) + ' Ambiguous terms')
    print('Ambiguous percentage: ' + str(round((counter / terms),4)))

def naive_tags(term_dict):
    bad_matches = 0
    for key in term_dict:
        if len(term_dict[key]) > 1:          
            max_value = max(term_dict[key].items(), key=operator.itemgetter(1))[0]
            total = 0
            for pos in term_dict[key]:
                total += term_dict[key][pos]
            bad_matches += (total - term_dict[key][max_value])
            term_dict[key] = {max_value: total}
            
    print("Total bad matches: " + str(bad_matches))
    print("Total terms: " + str(count_tags(term_dict)))
    print("Accuracy: " + str(1-(bad_matches/count_tags(term_dict))))
    return term_dict


analyse_tagged_text()


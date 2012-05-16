#!/usr/bin/env python

import subprocess
import sys
import os
import argparse
import cPickle as pickle
import ElectreTri

import random


directory = '/var/spool/samfilter/'
SENDMAIL = '/usr/sbin/sendmail'
SPAMC = '/usr/bin/spamc'

model_file = "samfilter.ets"

parser = argparse.ArgumentParser( version='samfilter 0.1' )
subparsers = parser.add_subparsers(help='commands', dest='command')

# filter
all_parser = subparsers.add_parser('filter', help='filter a single email in stdin')

# learn
app_parser = subparsers.add_parser('learn', help='make it learn from spam/ham learning set')
app_parser.add_argument('--hamdir', action='store', type=str, help='name of app to process')
app_parser.add_argument('--spamdir', action='store', type=str, help='name of app to process')

#args will contain the namespace with all the options
args = parser.parse_args()

if args.command == "filter":


    #mail_content = 'Subject: spam report'
    
    #mail_content += '\n' + sys.stdin.read()
    
    mail_content = sys.stdin.read()
    
    #send the mail content into spamc (spamc will redirect the content to spamd)
    spamc = subprocess.Popen([SPAMC, '-y'], stdin = subprocess.PIPE, stdout = subprocess.PIPE)
    spamc_analysis = spamc.communicate(mail_content)[0]
    
    #hit_list is a list of all the test names from spamassassin that have responded positive
    hit_list =  spamc_analysis.split(',')
    print hit_list
    
    #send the analysis to jglouis@samifis.be
    sendmail = subprocess.Popen([SENDMAIL, 'jglouis@samifis.be'], stdin = subprocess.PIPE)
    report = 'Subject: spam report\n'
    report += spamc_analysis
    
    sendmail.communicate(report)

elif args.command == "learn":
    print "learning..."
    try:
        #trying to load the ElectreTriSimple model from the file
        ets = pickle.load(open(model_file, "r+"))
        print "old model:"
        print ets
    except:
        #if the file cannot be loaded, then create a new one
        crits = [ElectreTri.Criterium("SPAMCOP"),
                 ElectreTri.Criterium("PYZOR"),
                 ElectreTri.Criterium("RAZOR"),
                 ElectreTri.Criterium("BAYES"),                 
                 ]
        
        lp1 = ElectreTri.LimitProfile("lp1") #limit profile separating HAM from SPAM
        categories = [ElectreTri.Category(1, name = "SPAM", lp_inf = lp1),
                      ElectreTri.Category(2, name = "HAM", lp_sup = lp1)
                      ]

        ets = ElectreTri.ElectreTriSimple(crits, categories)
    
    #build the alternatives from the learning set
    performance_table = ElectreTri.PerformanceTable(ets.crits, [])
    
    for file_name in os.listdir(args.spamdir):
        alt = ElectreTri.Alternative(file_name, ets.categories[0])
        performance_table.append_alternative(alt)
        """
        spamc = subprocess.Popen([SPAMC, '-y'], stdin = subprocess.PIPE, stdout = subprocess.PIPE)
        spamc_analysis = spamc.communicate(open(args.spamdir + '/' + file_name).read())[0].splitlines()
        """
        spamc_analysis = []
        for dummy in range(random.randint(1,4)):
            spamc_analysis.append(random.choice(['RCVD_IN_BL_SPAMCOP_NET','PYZOR_CHECK','RAZOR2_CHECK','BAYES_00','BAYES_99']))
        
        
        for crit in ets.crits: 
            test_score = 0.0
            for test_name in spamc_analysis:
                if crit.name in test_name:
                    if crit.name is 'BAYES':
                        #extract the last two digits and convert it into a percentage
                        test_score = int(test_name[-2:]) / 100.0
                        print test_score
                    else:
                        test_score = 1.0
                    
            performance_table.set_perf(alt, crit, test_score)
    
    for file_name in os.listdir(args.hamdir):
        alt = ElectreTri.Alternative(file_name, ets.categories[1])
        performance_table.append_alternative(alt)
        """
        spamc = subprocess.Popen([SPAMC, '-y'], stdin = subprocess.PIPE, stdout = subprocess.PIPE)
        spamc_analysis = spamc.communicate(open(args.hamdir + '/' + file_name).read())[0].splitlines()
        """
        
        spamc_analysis = []
        for dummy in range(random.randint(1,4)):
            spamc_analysis.append(random.choice(['RCVD_IN_BL_SPAMCOP_NET','PYZOR_CHECK','RAZOR2_CHECK','BAYES_00','BAYES_99']))
        
        for crit in ets.crits: 
            test_score = 0.0
            for test_name in spamc_analysis:
                if crit.name in test_name:
                    if crit.name is 'BAYES':
                        #extract the last two digits and convert it into a percentage
                        test_score = int(test_name[-2:]) / 100.0
                    else:
                        test_score = 1.0
                    
            performance_table.set_perf(alt, crit, test_score)


    print performance_table

    
    ets.learn_two_cat2(performance_table)
    print "new model:"
    print ets
    ets.save(model_file)
        

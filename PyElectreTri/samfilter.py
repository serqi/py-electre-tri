#!/usr/bin/env python

import subprocess
import sys
import argparse


directory = '/var/spool/samfilter/'
SENDMAIL = '/usr/sbin/sendmail'
SPAMC = '/usr/bin/spamc'



parser = argparse.ArgumentParser( version='samfilter 0.1' )
subparsers = parser.add_subparsers(help='commands', dest='command')

# filter
all_parser = subparsers.add_parser('filter', help='filter a single email in stdin')

# learn
app_parser = subparsers.add_parser('learn', help='make it learn from spam/ham learning set')
app_parser.add_argument('--hamdir', action='store', type=str, help='name of app to process')
app_parser.add_argument('--spamdir', action='store', type=str, help='name of app to process')

args = parser.parse_args()

print args

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
    print "learning"

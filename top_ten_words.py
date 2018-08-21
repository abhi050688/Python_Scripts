# -*- coding: utf-8 -*-
"""
Created on Sun Nov 05 18:07:35 2017

@author: Abhishek S
"""

filename="S:\Anaconda\Python_Scripts\mbox.txt"
file=open(filename,'r')
i=0
sndr=dict()
for line in file:
    line.strip()
    if line.startswith('From'):
        line=line.split()
        email=line[1]
        sender=email.split('@')[0]
        sndr[sender]=sndr.get(sender,0)+1
#print sndr
def r_dict(dict_to_rev):
    new_dict=dict([(v,k) for (k,v) in dict_to_rev.items()])
    return new_dict
r_sndr=r_dict(sndr)
r_sndr=r_sndr.items()
r_sndr.sort(reverse=True)
sorted_sndr=[(v,k) for k,v in r_sndr]
print 'Top ten senders are :'
for i in range(10):
    (sn,val)=sorted_sndr[i]
    print i+1,': ',sn.capitalize(),' with total emails sent: ',val

    
print 'Top ten sender are '
print sorted_sndr[:10]
    
#        i=i+1
#        if i>10:break
        



#!/usr/bin/env python

import sys,os
#import numpy as np
#import matplotlib.pyplot as plt
import argparse


def import_results(filename):
    results={"filename":filename}
    with open(filename) as file:
        for line in file:
            line=line.rstrip()
            if line=="" or "#" in line: continue
            unpack=line.split()
            variable=unpack[0]
            value=" ".join(unpack[1:])
            if value.isdigit(): #leave ints the same
                value=value
            elif value.replace('.','',1).isdigit(): #check if float in fancy way, doens't do negatives?
                value=f"%.2f"%(float(value))
            results[variable]=value

    return results


if __name__ == "__main__":

    #Read arguments
    parser = argparse.ArgumentParser(description="Make results tables from model arguments.txt file")
    parser.add_argument('-i',"--inputs", nargs='+', help='Inputs', required=True)
    parser.add_argument('-m',"--metrics", nargs='+', help='Inputs', required=True)
    parser.add_argument("-o","--output-file", default="results.txt", help="Ouput file")
    args=parser.parse_args()


    #Load the results
    results=[]
    for filename in args.inputs:
        results.append(import_results(filename))


    #Plot results
    print("\n")
    string=""
    for metric in args.metrics:
        string+=f" | %15s"%metric
    string+=" |"
    print(string)
    print('-' * (len(string)+1))
    for result in results:
        string=""
        for metric in args.metrics:
            if not metric in result:
                string+=f" | %15s"%"-"
            else:
                string+=f" | %15s"%result[metric]
        string+=" |"
        print(string)
    print("\n")
        

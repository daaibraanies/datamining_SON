# coding=<utf-8>
import findspark
import pyspark
import time
import itertools
from pyspark.sql import SQLContext
from pyspark import TaskContext
import datetime
import json
import sys
import math

findspark.find()
spark_context = pyspark.SparkContext()
sql_context = SQLContext(spark_context)
task_context = TaskContext()

filter_threshold = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]


def delete_unicode_symbol(storage):
    for k in storage.keys():
        for itemset in storage[k]:
            if isinstance(itemset, tuple):
                for item in itemset:
                    itemset = str(item)


def dump_file(filename, data, mode='w'):
    with open(filename, mode) as out:
        out.write("{}".format(data))

def save_itemset(filename,items,n_items,header=False,mode='w'):
    with open(filename,mode) as out:
        if header:
            out.write(header+'\n')
        for i,item in enumerate(items):
            if n_items == 1:
                out.write("('" + str(item) + "')")
                if i != len(items) - 1:
                    out.write(",")
            else:
                out.write(str(item))
                if i != len(items) - 1:
                    out.write(",")
        out.write('\n\n')

def merge_results(new_file,candidates_file,freq_file):
    candidates = None
    freqs = None
    with open(candidates_file,'r') as c:
        candidates = c.read()
    with open(freq_file,'r') as f:
        freqs =  f.read()
    with open(new_file,'w') as out:
        out.write(candidates)
        out.write(freqs)


def output_itemsets(filename, header, storage, mode='w'):
    with open(filename, mode) as out:
        out.write(header + '\n')
        for k in storage.keys():
            for i, itemset in enumerate(storage[k]):
                if k == 1:
                    out.write("('" + str(itemset) + "')")
                    if i != len(storage[k]) - 1:
                        out.write(",")
                else:
                    out.write(str(itemset))
                    if i != len(storage[k]) - 1:
                        out.write(",")
            out.write('\n\n')


def first_MR_phase(buckets, threshold, save_to, k=1):
    candidates = None
    if k == 1:
        candidates = buckets \
            .filter(lambda x: x[1] >= threshold) \
            .map(lambda x: (x[0]))
    else:
        candidates = buckets \
            .filter(lambda x: x[1] >= threshold) \
            .map(lambda x: (x[0]))

    if len(candidates.take(1)) > 0:
        if k == 1:
            save_itemset('intr_candidates',sorted(candidates.collect()),k,header="Candidates:",mode='w')
        else:
            save_itemset('intr_candidates',sorted(candidates.collect()),k,header=False,mode='a')

    return candidates


def second_MR_phase(buckets, candidates, support, save_to, k=1):
    candidate_list = candidates.collect()
    result = None

    if k == 1:
        result = buckets.filter(lambda x: x[0] in candidate_list) \
            .filter(lambda x: x[1] >= support) \
            .map(lambda x: (x[0]))
    else:
        result = buckets.filter(lambda x: x[0] in candidate_list) \
            .filter(lambda x: x[1] >= support) \
            .map(lambda x: (x[0]))

    if len(result.take(1)) > 0:
        if k == 1:
            save_itemset('intr_freq',sorted(result.collect()),k,header="Frequent Itemsets:",mode='w')
        else:
            save_itemset('intr_freq',sorted(result.collect()),k,header=False,mode='a')

    return result


def SON(buskets, support, input_file_path, output_file_path):
    number_of_chunks = buskets.getNumPartitions()
    chunk_threshold = math.ceil(support / number_of_chunks)
    candidates = {}
    frequent_items = {}
    k = 1

    counts = buskets.flatMap(lambda x: [(bsn, 1) for bsn in x[1]]) \
        .reduceByKey(lambda x, y: x + y)

    first = first_MR_phase(counts, chunk_threshold, candidates, k=k)
    second = second_MR_phase(counts, first, support, frequent_items, k=k)

    while len(first.take(1)) > 0:
        k += 1
        # Bottleneck need to use only items from prev stage
        #TODO: хранить кандидатов с предыдущего шага и использовать только те комбинации, которые все в кандидатах
        counts = buskets.flatMap(lambda x: [(bsn, 1) for bsn in itertools.combinations(sorted(x[1]), k)]) \
            .reduceByKey(lambda x, y: x + y)

        print("------------------------------------------------------------Stage: {}".format(k))
        local_time = time.time()
        first = first_MR_phase(counts, chunk_threshold, candidates, k=k)
        second = second_MR_phase(counts, first, support, frequent_items, k=k)
        print("------------------------------------------------------------Stage time: {}".format(time.time() - local_time))

    #output_itemsets(output_file_path, "Candidates:", candidates, mode='w')
    #output_itemsets(output_file_path, "Frequent Itemsets:", frequent_items, mode='a')
    merge_results(output_file_path,'intr_candidates','intr_freq')


if __name__ == "__main__":
    # Form baskets user:[businesses]
    start_time = time.time()

    baksets = spark_context.textFile(input_file_path) \
        .filter(lambda line: not line.endswith("_id")) \
        .map(lambda line: line.split(",")) \
        .filter(lambda line: len(line) > 1) \
        .groupByKey() \
        .map(lambda x: (x[0], list(set(x[1])))) \
        .filter(lambda x: len(x[1]) > filter_threshold)

    SON(baksets, support, input_file_path, output_file_path)
    print("Duration: {:.2f}".format(time.time() - start_time))
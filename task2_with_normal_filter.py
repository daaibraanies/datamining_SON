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

filter_threshold = 70
support = 50
input_file_path = 'large.csv'
output_file_path = 'results'


def delete_unicode_symbol(storage):
    for k in storage.keys():
        for itemset in storage[k]:
            if isinstance(itemset, tuple):
                for item in itemset:
                    itemset = str(item)


def dump_file(filename, data, mode='w'):
    with open(filename, mode) as out:
        out.write("{}".format(data))


def save_itemset(filename, items, n_items, header=False, mode='w'):
    with open(filename, mode) as out:
        if header:
            out.write(header + '\n')
        for i, item in enumerate(items):
            if n_items == 1:
                out.write("('" + str(item) + "')")
                if i != len(items) - 1:
                    out.write(",")
            else:
                out.write(str(item))
                if i != len(items) - 1:
                    out.write(",")
        out.write('\n\n')


def merge_results(new_file, candidates_file, freq_file):
    candidates = None
    freqs = None
    with open(candidates_file, 'r') as c:
        candidates = c.read()
    with open(freq_file, 'r') as f:
        freqs = f.read()
    with open(new_file, 'w') as out:
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


def first_MR_phase(buckets, threshold, k=1):
    candidates = None
    if k == 1:
        candidates = buckets \
            .filter(lambda x: x[1] >= threshold) \
            .map(lambda x: (x[0]))
    else:
        candidates = buckets \
            .filter(lambda x: x[1] >= threshold) \
            .map(lambda x: (x[0]))

    candidates = sorted(candidates.collect())

    if len(candidates) > 0:
        if k == 1:
            save_itemset('intr_candidates', candidates, k, header="Candidates:", mode='w')
        else:
            save_itemset('intr_candidates', candidates, k, header=False, mode='a')

    return candidates


def second_MR_phase(buckets, candidates, support, k=1):
    candidate_set = set(candidates)
    result = None

    if k == 1:
        result = buckets.filter(lambda x: x[0]) \
            .filter(lambda x: x[1] >= support) \
            .map(lambda x: (x[0]))
    else:
        result = buckets.filter(lambda x: x[0]) \
            .filter(lambda x: x[1] >= support) \
            .map(lambda x: (x[0]))

    result = sorted(result.collect())

    if len(result) > 0:
        if k == 1:
            save_itemset('intr_freq', result, k, header="Frequent Itemsets:", mode='w')
        else:
            save_itemset('intr_freq', result, k, header=False, mode='a')

    return result

def is_valid_candidate(itemset,prev_frequents,k):
    min_freq_occurances = k-1
    set_size = k - 1
    return True if len(
                    set(prev_frequents).intersection(set(itertools.combinations(itemset,set_size)))
                    ) >= min_freq_occurances else False

def SON(buskets, support, input_file_path, output_file_path):
    number_of_chunks = buskets.getNumPartitions()
    chunk_threshold = math.ceil(support / number_of_chunks)
    k = 1

    counts = buskets.flatMap(lambda x: [(bsn, 1) for bsn in x[1]]) \
        .reduceByKey(lambda x, y: x + y)

    first = first_MR_phase(counts, chunk_threshold, k=k)
    second = second_MR_phase(counts, first, support, k=k)

    while len(first) > 0:
        k += 1
        counts = None

        if k==2:
            counts = buskets.flatMap(lambda x: [(bsn, 1) for bsn in itertools.combinations(sorted(x[1]), k) if
                                                set(bsn).issubset(set(second))]) \
                .reduceByKey(lambda x, y: x + y)
        else:
            counts = buskets.flatMap(lambda x: [(bsn, 1) for bsn in itertools.combinations(sorted(x[1]), k) if
                                                len(set(second).intersection(set(itertools.combinations(bsn, k-1)))) >= 2])\
                .reduceByKey(lambda x, y: x + y)


        print("------------------------------------------------------------Stage: {}".format(k))
        local_time = time.time()

        first = first_MR_phase(counts, chunk_threshold, k=k)
        second = second_MR_phase(counts, first, support, k=k)

        print("------------------------------------------------------------Stage time: {}".format(time.time() - local_time))

    merge_results(output_file_path, 'intr_candidates', 'intr_freq')


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
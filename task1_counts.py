import findspark
import pyspark
import time
import itertools
from pyspark.sql import SQLContext
from pyspark import TaskContext
import datetime
import json
import sys

findspark.find()
spark_context = pyspark.SparkContext()
sql_context = SQLContext(spark_context)
task_context = TaskContext()

case_number = int(sys.argv[1])
support = int(sys.argv[2])
input_file_path = sys.argv[3]
output_file_path = sys.argv[4]


def dump_file(filename, data, mode='w'):
    with open(filename, mode) as out:
        out.write("{}".format(data))


def first_MR_phase(buckets, threshold, save_to, is_log=False, k=1):
    candidates = None
    if k == 1:
        candidates = buckets\
            .filter(lambda x: x[1] >= threshold) \
            .map(lambda x: (x[0]))
    else:
        candidates = buckets \
            .filter(lambda x: x[1] >= threshold) \
            .map(lambda x: (x[0]))

    if len(candidates.collect()) > 0:
        save_to[k] = "{}".format(sorted(candidates.collect()))

    if is_log:
        dump_file('combinations', candidates.collect())
    return candidates


def second_MR_phase(buckets, candidates, support, save_to, k=1):
    candedate_list = candidates.collect()
    result = None

    if k == 1:
        result = buckets.filter(lambda x: x[0] in candedate_list) \
            .filter(lambda x: x[1] >= support) \
            .map(lambda x: (x[0]))
    else:
        result = buckets.filter(lambda x: x[0] in candedate_list)\
            .filter(lambda x: x[1] >= support) \
            .map(lambda x: (x[0]))

    if len(result.collect()) > 0:
        save_to[k] = "{}".format(sorted(result.collect()))

    return result


def SON(buskets,support, input_file_path, output_file_path):
    number_of_chunks = buskets.getNumPartitions()
    chunk_threshold = support / number_of_chunks
    candidates = {}
    frequent_items = {}
    k = 1


    counts = buskets.flatMap(lambda x: [(bsn, 1) for bsn in x[1]]) \
            .reduceByKey(lambda x, y: x + y)

    first = first_MR_phase(counts, chunk_threshold, candidates, k=k)
    second = second_MR_phase(counts, first, support, frequent_items, k=k)

    while len(first.collect()) > 0:
        k += 1
        counts = buskets.flatMap(lambda x: [(bsn, 1) for bsn in itertools.combinations(sorted(x[1]), k)]) \
                .reduceByKey(lambda x, y: x + y)
        first = first_MR_phase(counts, chunk_threshold, candidates, k=k)
        second = second_MR_phase(counts, first, support, frequent_items, k=k)

    with open(output_file_path, 'w') as output:
        output.write('Candidates:\n')
        for k in candidates.keys():
            output.write(candidates[k])
            output.write('\n')
            output.write('\n')

        output.write('Frequent Itemsets:\n')
        for k in frequent_items.keys():
            output.write(frequent_items[k])
            output.write('\n')
            output.write('\n')


if __name__ == "__main__":
    # Form baskets user:[businesses]
    start_time = time.time()
    baksets = None
    if case_number == 1:
        baksets = spark_context.textFile(input_file_path) \
            .filter(lambda line: not line.endswith("_id")) \
            .map(lambda line: line.split(",")) \
            .filter(lambda line: len(line) > 1) \
            .groupByKey().map(lambda x: (x[0], list(set(x[1]))))

    else:
       baksets = spark_context.textFile(input_file_path) \
            .filter(lambda line: not line.endswith("_id")) \
            .map(lambda line: line.split(",")) \
            .filter(lambda line: len(line) > 1) \
            .groupByKey().map(lambda x: (x[1], list(set(x[0]))))

    SON(baksets,support, input_file_path, output_file_path)
    print("Duration: {:.2f}".format(time.time() - start_time))
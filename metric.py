#coding:utf-8
import sys, json

def calc_average_precision_at_k(labels, k):
    n = min(len(labels),k)
    labels = labels[:n]
    p = []
    p_cnt = 0
    for i in range(n):
        if labels[i]>0:
            p_cnt+=1
            p.append(p_cnt*1.0/(i+1))
    if p_cnt > 0:
        return sum(p)/p_cnt
    else:
        return 0.0

def calc_precision_at_k(labels, k):
    n = min(len(labels),k)
    labels = labels[:n]
    p_cnt = 0
    for i in range(n):
        if labels[i]>0:
            p_cnt+=1
    return p_cnt*1.0/n

def make_metric_dict():
    return { 'p@5':0, 'p@10':0, 'p@1':0, 'map@5':0, 'map@10':0, 'map@30':0 }

def main():
    metric_keys = [ 'p@1', 'p@5', 'p@10', 'map@5', 'map@10', 'map@30' ]
    filename = sys.argv[1]
    print "calc metric from %s" % filename
    f = file(filename)
    cnt = 0
    d = {} # for stat
    for line in f:
        try:
            step_labels = [ json.loads(labels) for labels in line.strip().split("\t") ]
        except:
            print line
            continue
        n = len(step_labels)
        for i in range(n):
            if not d.has_key(i):
                d[i] = make_metric_dict()
            d[i]['p@1'] += calc_precision_at_k(step_labels[i], 1)
            d[i]['p@5'] += calc_precision_at_k(step_labels[i], 5)
            d[i]['p@10'] += calc_precision_at_k(step_labels[i], 10)
            d[i]['map@5'] += calc_average_precision_at_k(step_labels[i], 5)
            d[i]['map@10'] += calc_average_precision_at_k(step_labels[i], 10)
            d[i]['map@30'] += calc_average_precision_at_k(step_labels[i], 30)
        cnt+=1
    f.close()
    n = len(d)
    print 'total_record_cnt=%d step_range=[0,%d]' % (cnt, n-1)
    for i in range(n):
        info = ["step=%d" % i]
        for key in metric_keys:
            info.append("%s=%0.2f" % (key, d[i][key] * 100.0/cnt))
        print " ".join(info)


if __name__=="__main__":
    main()


import csv

cnt = 0
id = {}
with open('periods_order.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt == 0:
            print(row)
        if cnt > 0:
            id[row[0]] = row[1]
        cnt = cnt + 1


cnt = 0
with open('../data_loader/classes_top200.csv', 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        if cnt == 0:
            print(row)
        if cnt > 0:
            period = row[3]
            print(id[period])
            #site, period = row[1].split('_')
            #print(row[2])
            # try:
            #     mm = id[period]
            #     #print(id[period])
            # except:
            #      print(period)
        cnt = cnt + 1
            #     #period_exception_list.append(period)


#exc_set = set(period_exception_list)
#for prd in exc_set:
#    print(prd)
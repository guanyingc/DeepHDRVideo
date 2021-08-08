import os
import numpy as np
from imageio import imread, imsave
import re

### IO Related ###
def make_file(f):
    if not os.path.exists(f):
        os.makedirs(f)
    #else:  raise Exception('Rendered image directory %s is already existed!!!' % directory)

def make_files(f_list):
    for f in f_list:
        make_file(f)

def empty_file(name):
    with open(name, 'w') as f:
        f.write(' ')

def read_list(list_path,ignore_head=False, sort=False):
    lists = []
    with open(list_path) as f:
        lists = f.read().splitlines()
    if ignore_head:
        lists = lists[1:]
    if sort:
        lists.sort(key=natural_keys)
    return lists

def split_list(in_list, percent=0.99):
    num1 = int(len(in_list) * percent)
    #num2 = len(in_list) - num2
    rand_index = np.random.permutation(len(in_list))
    list1 = [in_list[l] for l in rand_index[:num1]]
    list2 = [in_list[l] for l in rand_index[num1:]]
    return list1, list2

def write_string(filename, string):
    with open(filename, 'w') as f:
        f.write('%s\n' % string)

def save_list(filename, out_list):
    f = open(filename, 'w')
    #f.write('#Created in %s\n' % str(datetime.datetime.now()))
    for l in out_list:
        f.write('%s\n' % l)
    f.close()

def create_dirs(root, dir_list, sub_dirs):
    for l in dir_list:
        makeFile(os.path.join(root, l))
        for sub_dir in sub_dirs:
            makeFile(os.path.join(root, l, sub_dir))

#### String Related #####
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text) ]

def dict_to_string(dicts, start='\t', end='\n'):
    strs = '' 
    for k, v in sorted(dicts.items()):
        strs += '%s%s: %s%s' % (start, str(k), str(v), end) 
    return strs

def float_list_to_string(l):
    strs = ''
    for f in l:
        strs += ',%.2f' % (f)
    return strs

def insert_suffix(name_str, suffix):
    str_name, str_ext = os.path.splitext(name_str)
    return '%s_%s%s' % (str_name, suffix, str_ext)

def insert_char(mystring, position, chartoinsert):
    mystring = mystring[:position] + chartoinsert + mystring[position:] 
    return mystring  

def get_datetime(minutes=False):
    t = datetime.datetime.now()
    dt = ('%02d-%02d' % (t.month, t.day))
    if minutes:
        dt += '-%02d-%02d' % (t.hour, t.minute)
    return dt

def check_in_list(list1, list2):
    contains = []
    for l1 in list1:
        for l2 in list2:
            if l1 in l2.lower():
                contains.append(l1)
                break
    return contains

def remove_slash(string):
    if string[-1] == '/':
        string = string[:-1]
    return string

### Debug related ###
def check_div_by_2exp(h, w):
    num_h = np.log2(h)
    num_w = np.log2(w)
    if not (num_h).is_integer() or not (num_w).is_integer():
        raise Exception('Width or height cannot be devided exactly by 2exponet')
    return int(num_h), int(num_w)

def raise_not_defined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)


from collections import OrderedDict
import numpy as np

class Records(object):
    """
    Records->Train,Val->Loss,Accuracy->Epoch1,2,3->[v1,v2]
    IterRecords->Train,Val->Loss, Accuracy,->[v1,v2]
    """
    def __init__(self, records=None):
        if records == None:
            self.records = OrderedDict()
        else:
            self.records = records
        self.iter_rec = OrderedDict() # every iteration
        self.iter_avg_rec = OrderedDict() # every --disp_num iteration
        self.classes = ['loss', 'acc', 'err', 'ratio', 'psnr', 'lsnr', 'mse', 'ssim', 'range', 'edge']
        self.summary_classes = ['low', 'mid', 'high']

    def reset_iter(self):
        self.iter_rec.clear()

    def check_dict(self, a_dict, key, sub_type='dict'):
        if key not in a_dict.keys():
            if sub_type == 'dict':
                a_dict[key] = OrderedDict()
            if sub_type == 'list':
                a_dict[key] = []

    def update_iter(self, split, keys, values):
        self.check_dict(self.iter_rec, split, 'dict')
        for k, v in zip(keys, values):
            self.check_dict(self.iter_rec[split], k, 'list')
            self.iter_rec[split][k].append(v)

    def save_iter_record(self, split, epoch, iter_avg_rec, reset=True):
        self.check_dict(self.records, split, 'dict')
        for k in iter_avg_rec:
            #for k in self.iter_rec[s].keys():
            self.check_dict(self.records[split], k, 'dict')
            self.check_dict(self.records[split][k], epoch, 'list')
            self.records[split][k][epoch].append(iter_avg_rec[k])
        if reset: 
            self.reset_iter()

    def insert_record(self, split, key, epoch, value):
        self.check_dict(self.records, split, 'dict')
        self.check_dict(self.records[split], key, 'dict')
        self.check_dict(self.records[split][key], epoch, 'list')
        self.records[split][key][epoch].append(value)

    def cvt_val_to_str(self, val, cls='err'):
        if cls in ['psnr', 'range', 'low', 'high', 'lsnr', 'mid']:
            prec = '%.2f'
        elif cls in ['acc', 'ssim', 'ratio'] :
            prec = '%.4f'
        else:
            prec = '%.5f'
        return prec % val

    def prev_rec_to_string(self, split, epoch):
        rec_strs = ''
        defined_keys = ['prev']
        for c in self.classes:
            strs = ''
            for k in self.iter_avg_rec:
                if (c in k.lower()) and k.lower()[:4] in defined_keys: # records in prev stage
                    value_str = self.cvt_val_to_str(self.iter_avg_rec[k], cls=c)
                    strs += '%s: %s ' % (k[5:], val_str)
            if strs != '':
                rec_strs += '\t [%s] %s\n' % ('prev'.upper(), strs)
        return rec_strs

    def iter_rec_to_string(self, split, epoch):
        self.iter_avg_rec = self.iter_rec_to_dict(split)
        rec_strs = self.prev_rec_to_string(split, epoch)
        for c in self.classes:
            strs = ''
            for k in self.iter_avg_rec:
                if (c in k.lower()) and k.lower()[:5] != 'prev_':
                    #value_str = self.cvt_val_to_str(c) % self.iter_avg_rec[k]
                    value_str = self.cvt_val_to_str(self.iter_avg_rec[k], cls=c)
                    strs += '%s: %s ' % (k, value_str)
            if strs != '':
                rec_strs += '\t [%s] %s\n' % (c.upper(), strs)
        self.save_iter_record(split, epoch, self.iter_avg_rec)
        return rec_strs

    def iter_rec_to_dict(self, split):
        if split not in self.iter_rec:
            return OrderedDict()
        iter_avg_rec = OrderedDict()
        for k in self.iter_rec[split].keys():
            iter_avg_rec[k] = np.mean(self.iter_rec[split][k])
        return iter_avg_rec

    def epoch_rec_to_string(self, split, epoch):
        rec_strs = ''
        for c in self.classes + self.summary_classes:
            strs = ''
            for k in self.records[split].keys():
                if (c in k.lower()) and (epoch in self.records[split][k].keys()):
                    value_str = self.cvt_val_to_str(np.mean(self.records[split][k][epoch]), c)
                    strs += '%s: %s| ' % (k, value_str)
            if strs != '':
                rec_strs += '\t [%s] %s\n' % (c.upper(), strs)
        return rec_strs

    def record_to_dict_of_array(self, splits, epoch=-1, intv=1):
        if len(self.records) == 0: return {}
        if type(splits) == str: splits = [splits]

        dict_of_array = OrderedDict()
        for split in splits:
            for k in self.records[split].keys():
                y_array, x_array = [], []
                if epoch < 0:
                    for ep in self.records[split][k].keys():
                        y_array.append(np.mean(self.records[split][k][ep]))
                        x_array.append(ep)
                else:
                    if epoch in self.records[split][k].keys():
                        y_array = np.array(self.records[split][k][epoch])
                        x_array = np.linspace(intv, intv*len(y_array), len(y_array))
                dict_of_array[split[0] + split[-1] + '_' + k]      = y_array
                dict_of_array[split[0] + split[-1] + '_' + k+'_x'] = x_array
        return dict_of_array

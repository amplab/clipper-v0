
from __future__ import print_function
import os, shutil, sys, datetime
from subprocess import call, Popen, PIPE
from scipy.io import wavfile
import numpy as np
import json



def predict(input_file, true_label, model_path, tmp_fname):

    base_path = "/dev/shm/clipper-tmp"
    # model_path = os.path.expanduser("~/timit/timit_model")
    fname = tmp_fname
    # wavfname = base_path + '/' + fname + '.wav'
    wavfname = base_path + "/" + fname + ".wav"
    # print(input_file[-4:])
    assert input_file[-4:] == ".wav"
    copy_start = datetime.datetime.now()

    # First put it in the in-mem file system, because accessing
    # home dir over NFS is *really* slow
    shutil.copy(input_file, wavfname)
    tl_fname = base_path + "/" + fname + ".lab"
    shutil.copy(true_label, tl_fname)
    copy_end = datetime.datetime.now()
    # print("copy latency: %f ms" % ((copy_end - copy_start).total_seconds() * 1000))

    start = datetime.datetime.now()
    mfccfname = base_path +'/' + fname + '.mfc'
    # print("converting wav to mfcc")
    # print(wavfname, mfccfname)

    hc_start = datetime.datetime.now()
    call(['HCopy', '-C', os.path.expanduser('~/timit/timit_tools/wav_config'), wavfname, mfccfname])
    hc_end = datetime.datetime.now()
    # print("hcopy latency: %f ms" % ((hc_end - hc_start).total_seconds() * 1000))
    # print("making prediction")
    # lst = (['HVite',
    #      '-p 2.5',
    #      '-s 5.0',
    #      '-w %s/wdnetbigram' % model_path,
    #      '-H %s/hmm_final/hmmdefs' % model_path,
    #      '-i %s/outtrans.mlf' % base_path,
    #      '-o ST',
    #      '%s/dict' % model_path,
    #      '%s/phones' % model_path,
    #      mfccfname])
    inf_start = datetime.datetime.now()
    cmd_str_template = ("HVite -s 5.0 -p 2.5 -w %(mp)s/wdnetbigram "
                        "-H %(mp)s/hmm_final/hmmdefs -i %(bp)s/%(fn)s.mlf "
                        "-o ST %(mp)s/dict %(mp)s/phones %(mfccfname)s")

    cmd_str = cmd_str_template % { 'mp': model_path,
                                   'bp': base_path,
                                   'mfccfname': mfccfname,
                                   'fn': fname
                                  }

    p = call(cmd_str, shell=True)
    inf_end = datetime.datetime.now()
    # print("hvite latency: %f ms" % ((inf_end - inf_start).total_seconds() * 1000))
    end = datetime.datetime.now()
    # print("eval_latency: %f ms" % ((end - start).total_seconds() * 1000))

    proc = Popen('HResults %(mp)s/phones %(bp)s/%(fn)s.mlf'
         % { 'mp': model_path,
             'tl': tl_fname,
             'bp': base_path,
             'fn': fname
             }, shell=True, stdout=PIPE, stderr=PIPE)
    (output, err) = proc.communicate()
    # (correct, inserts, total) = get_results(output)
    return get_results(output)
    # print("correct: %f, accuracy: %f" % (correct / float(total), (correct - inserts) / float(total)))


def get_results(output):
    '''
        Parse the results of HResult and extract the accuracy and number
        of correct phonemes
    '''
    total = None
    correct = None
    inserts = None
    for l in output.split("\n"):
        if l[:4] == "WORD":
            entries = l.split(" ")
            # print(entries)
            correct = int(entries[3].split("=")[1].strip(","))
            inserts = int(entries[6].split("=")[1].strip(","))
            total = int(entries[7].split("=")[1].strip("]"))
            # print(correct, inserts, total)
    #         acc = float(entries[2].split("=")[1].strip(","))
    #         # print("corr: %f, acc: %f" % (corr, acc))
    return (correct, inserts, total)


def get_data_for_user(user_dir):
    samples = [w[:-4] for w in os.listdir(user_dir) if w[-4:] == ".wav"]
    permuted_samples = np.random.permutation(samples)
    # print(samples, permuted_samples)
    return [os.path.join(user_dir, p) for p in permuted_samples]



# def add_model_results(m1, m2):
#     return ModelResults(np.add(m1.corr, m2.corr),
#                         np.add(m1.inserts, m2.inserts),
#                         np.add(m1.num, m2.num))
#
#     
# def add_users(u1, u2):
#     return UserResults(add_model_results(u1.learned, u2.learned),
#                        add_model_results(u1.general, u2.general),
#                        add_model_results(u1.dialect, u2.dialect))
#

class AnytimeResults:
    def __init__(self, c, i, n):
        assert len(c) == len(i) and len(i) == len(n)
        self.corr = np.array(c)
        self.inserts = np.array(i)
        self.num = np.array(n)
    
    def acc(self):
        return (self.corr - self.inserts)/self.num.astype('float64')

    def acc_string(self):
        acc_str = ",".join(['{:.3f}'.format(x) for x in self.acc()]) + "\n"
        return acc_str


class ModelResults:
    def __init__(self, c, i, n):
        assert len(c) == len(i) and len(i) == len(n)
        self.corr = c
        self.inserts = i
        self.num = n

    def acc(self, max_index = -1):
        if max_index < 0:
            max_index = len(self.corr) - 1
        max_index += 1 # because the range is non-inclusive
        corr_total = np.sum(self.corr[:max_index])
        inserts_total = np.sum(self.inserts[:max_index])
        num_total = np.sum(self.num[:max_index])
        return (corr_total - inserts_total)/float(num_total)

    def sample_acc(self, index):
        assert index >= 0 and index < len(self.corr)
        return (self.corr[index] - self.inserts[index])/float(self.num[index])

    def get_index(self, index):
        assert index >= 0 and index < len(self.corr)
        return (self.corr[index], self.inserts[index], self.num[index])

# class UserResults:
#     def __init__(self, uname, learned, general, dialect, learned_model):
#         self.learned = learned
#         self.general = general
#         self.dialect = dialect
#         self.uname = uname
#         self.learned_model = learned_model
#
#     def acc_per_iter(self):
#         learned_iter_accs = []
#         general_iter_accs = []
#         dialect_iter_accs = []
#         for i in range(len(self.learned.corr)):
#             learned_iter_accs.append(self.learned.sample_acc(i))
#             general_iter_accs.append(self.general.sample_acc(i))
#             dialect_iter_accs.append(self.dialect.sample_acc(i))
#         learned_iter_accs = ['{:.3f}'.format(x) for x in learned_iter_accs]
#         general_iter_accs = ['{:.3f}'.format(x) for x in general_iter_accs]
#         dialect_iter_accs = ['{:.3f}'.format(x) for x in dialect_iter_accs]
#         return(learned_iter_accs, general_iter_accs, dialect_iter_accs)
#
            
    

def eval_user(dr, uname, tmp_fname):

    samples = get_data_for_user(os.path.join(dr, uname))
    holdout_sample = samples[-1]
    samples = samples[:-1] # remove last sample

    # print(samples)
    model_dir_path = "/crankshaw-local/timit_models/models"
    gen_model_path = os.path.join(model_dir_path, "undersampled_general_model")
    dr_models = [os.path.join(model_dir_path, "dr%d_model" % d) for d in range(1,9)]
    gen_model_path = os.path.join(model_dir_path, "undersampled_general_model")
    # models = [gen_model_path, dr_models[0]]
    models = dr_models + [gen_model_path]
    model_results = {}
    for m in models:
        corrs = []
        inserts = []
        nums = []
        for s in samples:
            (c, i, n) = predict(s + ".wav", s + ".lab", m, tmp_fname)
            corrs.append(c)
            inserts.append(i)
            nums.append(n)
        model_name = os.path.basename(m)
        model_results[model_name] = ModelResults(corrs, inserts, nums)

    # online learning algo: rank models by accuracy
    ranked_models = anytime_model_ranking(model_results)
    ranked_model_paths = [os.path.join(model_dir_path, m) for m in ranked_models]
    print(ranked_model_paths)

    # pick random order to drop models
    arrival_order = np.random.permutation(len(model_results))

    # eval accuracy
    anytime_results = eval_anytime_acc(holdout_sample, ranked_model_paths, arrival_order, tmp_fname)

    return anytime_results.acc_string()
    # print("learned_acc: %f, general_acc: %f, dialect_acc: %f" % (learned_acc, general_acc, dialect_acc))
    # print("user: %s, most correct: %s, most accurate: %s" % (uname, max_corr_model, max_acc_model))

def eval_anytime_acc(sample, ranked_models, arrival_order, tmp_fname):
    '''
        ranked_models is a list of model paths in order from best to worst.
        arrival_order is a permutation of the indices of the ranked models
        indicating the order that the ranked_models arrive in.
        To find the anytime accuracy for k models, we take the first k entries
        in the arrival order as the available models. The best one is
        the lowest value in that sublist.
    '''
    corr = []
    inserts = []
    total = []
    for i in range(1, len(arrival_order)):
        available_models = arrival_order[:i]
        best_model = np.min(available_models)
        (c, i, n) = predict(sample + ".wav", sample + ".lab", ranked_models[best_model], tmp_fname)
        corr.append(c)
        inserts.append(i)
        total.append(n)
    return AnytimeResults(corr, inserts, total)



# def eval_heldout_accuracy(sample, models, tmp_fname):
#     corr = []
#     inserts = []
#     total = []
#     for m in models:
#         (c, i, n) = predict(sample + ".wav", sample + ".lab", m, tmp_fname)
#         corr.append(c)
#         inserts.append(i)
#         total.append(n)
#     return ModelResults(corr, inserts, total)


# def eval_model_acc(model_results, learned_model):
#     corr = []
#     inserts = []
#     total = []
#     for idx in range(len(learned_model)):
#         m = learned_model[idx]
#         (c, i, n) = model_results[m].get_index(idx)
#         corr.append(c)
#         inserts.append(i)
#         total.append(n)
#     return ModelResults(corr, inserts, total)


def anytime_model_ranking(model_results):
    # always start with the general model
    learned_model = ["undersampled_general_model"]
    accs = []
    model_names = []
    for k in model_results:
        accs.append(model_results[k].acc())
        model_names.append(k)
    model_rank = np.flipud(np.argsort(np.array(accs)))
    ranked_models = []
    for i in model_rank:
        ranked_models.append(model_names[i])
    # print(model_names)
    # print(model_rank)
    print(ranked_models)
    # print(model_results)
    return ranked_models
    # print(model_rank)
    # return (np.array(model_names)[model_rank]).tolist()

# def find_learned_model(model_results, num_samples):
#     # always start with the general model
#     learned_model = ["undersampled_general_model"]
#     for i in range(1, num_samples):
#         best_model = ""
#         best_acc = 0.0
#         for k in model_results:
#             # we want the accuracy up to the previous point to
#             # tell us what model to use for this point
#             acc = model_results[k].acc(max_index = (i - 1))
#             if acc > best_acc:
#                 best_acc = acc
#                 best_model = k
#         learned_model.append(best_model)
#     return learned_model

# def dr_model(dr, model_path):
#     return os.path.join(model_path, "%s_model" % dr)

# from json import JSONEncoder
# class MyEncoder(JSONEncoder):
#     def default(self, o):
#         return o.__dict__    

def main(proc_num, min_dr, max_dr):
    model_dir_path = "/crankshaw-local/timit_models/models"
    test_data_path = "/crankshaw-local/timit_data/timit/test"
    dialects = ["dr%d" % i for i in range(min_dr, max_dr + 1)]
    iii = 0
    tmp_fname = "sample_%d" % proc_num
    print(tmp_fname)
    with open("timit_anytime_%d_to_%d.json" % (min_dr, max_dr), 'w') as rf:
        # rf.write("[\n")
        for d in dialects:
            print(d)
            test_dr_dir = os.path.join(test_data_path, d)
            users = [f for f in os.listdir(test_dr_dir) if os.path.isdir(os.path.join(test_dr_dir, f))]
            for u in users:
                # don't write a comma before the first entry

                # if iii > 0:
                #     rf.write(",\n")
                print(d, u, iii)
                rf.write(eval_user(test_dr_dir, u, tmp_fname))
                # json.dump(user_res, rf, cls=MyEncoder)
                iii += 1
            #     if iii > 3:
            #         break
            # break
        # rf.write("\n]")

if __name__=='__main__':
    proc_num = int(sys.argv[1])
    (min_dialect,max_dialect) = tuple([int(m) for m in sys.argv[2].split(":")])
    print("processing drs %d to %d" % (min_dialect, max_dialect))
    main(proc_num, min_dialect, max_dialect)

    print("\n\nPROCESS %d FINISHED" % proc_num)
















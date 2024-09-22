import os
import math
import morfessor

def log_func(x):
    return int(round(math.log(x + 1, 2)))
io = morfessor.MorfessorIO()

dir = '/home/connor/2024/wordlists/'
files = [os.path.join(dir, file) for file in os.listdir(dir)]
count = 1

if __name__ == '__main__':
    for f in files:
        lang = os.path.splitext(os.path.basename(f))[0].split('_')[0]
        train_data = list(io.read_corpus_file(f))
        model = morfessor.BaselineModel()
        model.load_data(train_data, count_modifier=log_func)
        model.train_batch()
        seg_file_name = f"/home/connor/2024/segmodels2/{lang}_model.bin"
        os.system(f'echo "Starting {count}/24"')
        io.write_binary_model_file(seg_file_name, model)
        count += 1
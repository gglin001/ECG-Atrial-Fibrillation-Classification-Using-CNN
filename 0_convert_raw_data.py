import glob
import os
import shutil


def convert_to_csv():
    fps = glob.glob('aftdb/*/*.hea')
    for fp in fps:
        # fp_name = fp.split('/')[-1].split('.')[0]
        fp_name = fp.split('.')[0]
        to_save = fp.replace('.hea', '.csv')
        cmd = 'rdsamp -r %s -c -H -f 0 -t 60 -v -pd > %s' % (fp_name, to_save)
        print(cmd)
        os.system(cmd)


def move_csv_to_folder():
#    for dir_name in ['train_set', 'test_set']:
#         if not os.path.exists(dir_name):
#             os.mkdir(dir_name)

    fps = glob.glob('aftdb/*/*.csv')
    for fp in fps:
        if 'test' in fp:
            move_to = 'test_set/%s' % fp.split('/')[-1]
            shutil.move(fp, move_to)
        elif 'learning' in fp:
            move_to = 'train_set/%s' % fp.split('/')[-1]
            shutil.move(fp, move_to)


if __name__ == "__main__":
    convert_to_csv()
    move_csv_to_folder()

from pyrouge import Rouge155
import os
from shutil import copyfile, rmtree
import cPickle as pickle
import tempfile
import sys

def get_scores(model_name):
    scores = []
    base_path = "/home/ml/mgrena/mod-target/pgn/pretrained_model_tf1.2.1/mod-target1375"
    decoded_path = os.path.join(base_path, "decoded")
    reference_path = os.path.join(base_path, "reference")

    if len(os.listdir(decoded_path)) != len(os.listdir(reference_path)):
        raise ValueError("Number of reference summaries and decoded summaries do not match")

    num_articles = len(os.listdir(reference_path))
    print(num_articles)
    if not os.path.exists("decoded_tmp"):
        os.mkdir("decoded_tmp")
    if not os.path.exists("reference_tmp"):
        os.mkdir("reference_tmp")
    if not os.path.exists("temp-files"):
        os.mkdir("temp-files")

    # Hacky housecleaning. Pyrouge stores these massive tmp files and it can't be turned off. So we store them here
    tempfile.tempdir = os.path.join(os.getcwd(), "temp-files")

    for i in range(0, num_articles):
        # File names
        decoded_filename = str(i).rjust(6, '0') + "_decoded.txt"
        reference_filename = str(i).rjust(6, '0') + "_reference.txt"

        # Copy files over to temp folder
        copyfile(os.path.join(decoded_path, decoded_filename), os.path.join("decoded_tmp/", decoded_filename))
        copyfile(os.path.join(reference_path, reference_filename), os.path.join("reference_tmp/", reference_filename))

        # ROUGE object
        r = Rouge155()
        r._system_dir = 'decoded_tmp/'
        r._model_dir = 'reference_tmp/'

        r.system_filename_pattern = '(\d+)_decoded.txt'
        r.model_filename_pattern = '#ID#_reference.txt'

        output = r.convert_and_evaluate()
        output_dict = r.output_to_dict(output)

        essential_keys = ['rouge_1_f_score', 'rouge_2_f_score', 'rouge_3_f_score', 'rouge_l_f_score']
        essential_dict = {key: output_dict[key] for key in essential_keys}
        scores.append(essential_dict)

        # Remove temp files
        os.unlink("decoded_tmp/" + decoded_filename)
        os.unlink("reference_tmp/" + reference_filename)

    # Pickle final results
    pickle_out = open(model_name + ".pic", "wb")
    pickle.dump(scores, pickle_out)
    pickle_out.close()

    # Delete excessive log files
    print("Removing temp files")
    rmtree(path="temp-files")
    rmtree(path="decoded_tmp/")
    rmtree(path="reference_tmp/")

if len(sys.argv) != 2:
    raise TypeError("Enter the name of the model as an argument")

get_scores(sys.argv[1])

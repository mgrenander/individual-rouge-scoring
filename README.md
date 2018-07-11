# Individual ROUGE Scoring
This script is quick hack that will compute individual ROUGE scores across a dataset.
The intended usage is to perform some data analysis on the individual article-summary ROUGE scores, such as viewing how scores vary under some guiding metric.


Usage is as follows:

``
python get_scores.py <name of model>
``

After running, the script will output a pickle file containing a list of dictionaries. Each dictionary file contains the ROUGE scores one article-summary pair.

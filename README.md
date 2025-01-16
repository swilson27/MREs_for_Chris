# MREs_for_Chris

I've based this repository on your template at https://github.com/navis-org/pymaid_template .

Within this repo are 3 python scripts (MRE{1/2/3}.py). Descriptions are found in the MRE_description.txt file

These 3 scripts/MREs are condensed down from 3 larger scripts (not attached here in full). Those 3 scripts all take left/right paired neurons in Seymour from a local csv file (attached here) and respectively run NBLAST, SyNBLAST, and cosine similarity within each pair (i.e. left and right neuron comparisons). The scripts are thus aiming to identify morphological or connectivity discrepancies between left/right paired neurons.

The code for the N/SyNBLAST derived from a script you wrote (when I was trying to do this for NBLAST: https://gist.github.com/clbarnes/920d1ac533bf76922f036a8b564df5ed)
Once I have these issues of the MREs fixed, I'd been planning to condense the 3 scripts into one (which makes sense from the view of only having the laborious fetching of neurons once)

As said over whatsapp, unfortunately I had issue with getting the neurons as zipped SWC files (which was due to be done via the attached 'get_neurons.py' script, utilising 'paired_neurons.csv'). I can't understand why it's failing to write the SWC files on the 3rd neuron, but described this error in the script.

The MREs instead load the neurons directly, and would require credentials (as 'seymour.json' in the credentials folder, as per the layout of your 'pymaid_template' repo). As you'd suggested you still have access to Seymour, hopefully that won't be a problem. If not, please let me know and I'll try again to fix the SWC issue

The MREs also rely on 'CNS_landmarks.csv' which is likewise attached

The virtual environment I've been using is in requirements.txt

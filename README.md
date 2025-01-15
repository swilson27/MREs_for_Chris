# MREs_for_Chris

I've based this repository on your template at https://github.com/navis-org/pymaid_template .

Within this repo are 3 separate MREs (named numerically). These are condensed down from 3 larger scripts (not attached here in full). Those 3 scripts 
all take left/right paired neurons in Seymour from a local csv file (attached here) and respectively run NBLAST, SyNBLAST, and cosine similarity within each pair (i.e. left and right neuron comparisons). The scripts are thus aiming to identify discrepancies between left/right paired neurons.

The code for the N/SyNBLAST derived from a script you wrote (when I was trying to do this for NBLAST: https://gist.github.com/clbarnes/920d1ac533bf76922f036a8b564df5ed)
Once I have the below issues fixed, I'd been planning to condense the 3 scripts into one (which makes sense from the view of only having the laborious fetching of neurons once)

I have additionally attached the left and right neurons, as zipped SWC files. For interest, these were obtained via the 'get_neurons.py' script, which relies on the 'paired_neurons.csv' file. 

the MREs rely on 'CNS_landmarks.csv' which is likewise attached



### First use

```sh
# Pick a name for your project
PROJECT_NAME="my_project"

# Clone this template, then change directory into it
git clone https://github.com/navis-org/pymaid_template.git "$PROJECT_NAME"
cd "$PROJECT_NAME"

# Delete the template's git history and license
rm -rf .git/ LICENSE
# Initialise a new git repo
git init
# Commit the existing files so you can track your changes
git add .
git commit -m "Template from navis-org/pymaid_template"

# Ensuring that you are using a modern version of python (3.9, here), create and activate a virtual environment
python3.9 -m venv --prompt "$PROJECT_NAME" venv
source venv/bin/activate
# use `deactivate` to deactivate the environment

# Install the dependencies
pip install -r requirements.txt

# Make your own credentials file
cp credentials/example.json credentials/credentials.json
```

Then edit the credentials file to access your desired CATMAID server, instance, and user.

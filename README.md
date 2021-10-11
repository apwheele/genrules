# Using a Genetic Algorithm to Identify Co-morbidities

This github project is for the NICHD’s *Decoding Maternal Morbidity Data Challenge*.

It uses a novel genetic algorithm to identify co-morbidities associated with a particular outcome (here maternal morbidities), by optimizing the risk ratio given different categorical inputs. 

Use notes:

 - To be replicated, all scripts need to be run from the root folder
 - The csv data for the nuMoM2b study (not uploaded to github), needs to be located at the `data/nuMoM2b.csv` to replicate the findings.
 - This uses python 3.7, and the libraries specified in `requirements.txt`. See that file for notes on creating your own conda environment to replicate.
 - The genetic algorithm does have stochastic elements, so your results may differ slightly from my results

The folder `/tech_docs` has more technical documentation on the genetic algorithm, but also note the source code for the functions is 100% provided in `/src/genrules.py`. 

The jupyter notebook `Example1_genrules.ipynb` provides examples of the base algorithm to identify particular comorbidities. To run this notebook locally, you can use the command:

    jupyter nbconvert --to notebook --execute Example1_genrules.ipynb --output Example1_genrules.ipynb

Or if you prefer to browse html output, you could use

    jupyter nbconvert --execute Example1_genrules.ipynb --to html

If you have any questions, please feel free to contact me,

Andrew Wheeler, PhD

awheeler29@gsu.edu

https://andrewpwheeler.com/
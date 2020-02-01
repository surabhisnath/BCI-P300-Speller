# BCI-P300-Speller

This repo contains code for determining the character in a P300 Speller. This is based on the concept of P300 ERP potential that arises due to an oddball stimulus. The speller serves as a vital communication tool for ALS or paralytic patients. We collected the 8 channel EEG data from an online source. Data was pre-processed to remove noise following which we applied machine learning classifiers (Linear discriminant Analysis worked best for us) to classify the data into target or non target. Using this, we found the mode row and mode column, intersection of those gave us our letter. We were able to achieve nearly 85% accuracy in correctly predicting the character.

https://www.frontiersin.org/articles/10.3389/fnhum.2013.00732/full

Data Source: http://bnci-horizon-2020.eu/database/data-sets
Data Description: https://lampx.tugraz.at/~bci/database/008-2014/description.pdf

This work was done under the SRIP IIT Gandhinagar, 2018

# Data Labeling

Three techniques for data labeling on traffic accident risks are presented below. 

## Ruled-based voting ensemble

A subset of observations is manually labeled by defining attribute-specific threshold values and establishing value ranges associated with varying penalty scores. These rules allowed us to construct interpretable labeling criteria for initial class assignment. A semi-supervised approach that combines our manual rules with a rule-based voting ensemble is employed to extend labels to the full dataset. Each labeling rule acted as a weak classifier, and final class assignments for the unlabeled data were determined through majority voting. This hybrid strategy leveraged domain knowledge and minimal manual effort to generate a complete and consistent labeling across the dataset.

## Fuzzy sets

Following the guidelines of fuzzy sets, the most relevant variables (13), the associated fuzzy sets for each variable, and then the membership functions for all fuzzy
sets (60) are created. Also, all fuzzy rules (55) to relate inputs and outputs. Regarding membership functions, a triangular function for all fuzzy sets is choosen. This function
establishes that the boundary values obtain the lowest membership values, and the mean value obtains the highest. Considering that the number of rules for a fuzzy inference system (FIS) with this number of attributes could easily reach hundreds or thousands, a FIS that consists of single-input and single-output (SISO) subsystems is designed, where the input is each attribute and the output is the penalty value for that input. Those penalty values are used later to calculate the risk level for that observation. 

## Fuzzy clustering

Fuzzy c-means (FCM) use different values for its hyperparameters, (2, 3, 4, and 5) for fuzziness factor, 0.005 for error, and 1,000 for number of iterations. 

Configurations for FCM

|#|Fuzziness|Error|Number of iterations|
|-|---|---|---|
|1|2|0.005|1,000|
|2|3|0.005|1,000|
|3|4|0.005|1,000|
|4|5|0.005|1,000|

# Publication

If you use POLIDriving in your research, please cite it as follows.

@article{marcillo2024polidriving,<br>
  title={POLIDriving: A Public-Access Driving Dataset for Road Traffic Safety Analysis},<br>
  author={Marcillo, Pablo and Arciniegas-Ayala, Cristian and Valdivieso Caraguay, {\'A}ngel Leonardo and Sanchez-Gordon, Sandra and Hern{\'a}ndez-{\'A}lvarez, Myriam},<br>
  journal={Applied Sciences},
  volume={14},<br>
  number={14},<br>
  pages={6300},<br>
  year={2024},<br>
  publisher={MDPI}<br>
}

# Downloads

The size of POLIDriving is about 150 MB.

# Contact

For questions or suggestions, please contact pablo.marcillo@epn.edu.ec or pablomarcillolara@gmail.com

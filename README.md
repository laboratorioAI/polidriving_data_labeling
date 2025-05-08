# Data Labeling

Three techniques for data labeling on traffic accident risks are presented below. 

## Ruled-based voting ensemble

A subset of observations is manually labeled by defining attribute-specific threshold values and establishing value ranges associated with varying penalty scores. These rules allowed us to construct interpretable labeling criteria for initial class assignment. A semi-supervised approach that combines our manual rules with a rule-based voting ensemble is employed to extend labels to the full dataset. Each labeling rule acted as a weak classifier, and final class assignments for the unlabeled data were determined through majority voting. This hybrid strategy leveraged domain knowledge and minimal manual effort to generate a complete and consistent labeling across the dataset.

## Fuzzy sets

Following the guidelines of fuzzy sets, we created the most relevant linguistic variables (13 in total), the associated fuzzy sets for each linguistic variable, and then the membership functions for all fuzzy
sets. Finally, we determined the fuzzy rules to relate inputs and outputs. Regarding membership functions, we chose a triangular function for all fuzzy sets. This function
establishes that the boundary values obtain the lowest membership values, and the mean value obtains the highest.. Considering that the number of rules for a fuzzy inference system (FIS) with this number of attributes could easily reach hundreds or thousands, a FIS that consists of single-input and single-output (SISO) subsystems is designed, where the input is each attribute and the output is the penalty value for that input. Those penalty values are used later to calculate the risk level for that observation. 

## Fuzzy clustering

Fuzzy c-means (FCM) with different values (2, 3, 4, and 5) for the fuzziness factor, 0.005 for error, and 1,000 for number of iterations. 



Table 3 presents the configurations used in clustering and their results. According to this table, the minority classes are high and very high. In the case of configuration #1, all classes are quite balanced; however, for the remaining configurations, the number of observations in minority classes is practically zero, except for configuration #2, where the number of observations labeled as high is somewhat numerous. Figure 3 presents the label distribution for some of the most relevant attributes for all configurations.

## Data file format

Data files contain the following attributes.

|#|Attribute|Class|Units|Data source|
|-|---|---|---|---|
|1|time|Timestamp||Vehicle data|
|2|speed|Numeric|km/h|Vehicle data|
|3|revolutions per minute|Numeric|rpm|Vehicle data|
|4|acceleration|Numeric|m/s2|Vehicle data|
|5|throttle position|Numeric|%|Vehicle data|
|6|engine temperature|Numeric|C|Vehicle data|
|7|system voltage|Numeric|volts|Vehicle data|
|8|distance traveled|Numeric|km|Vehicle data|
|9|engine load value|Numeric|%|Vehicle data|
|10|latitude|Numeric||Vehicle data|
|11|longitude|Numeric||Vehicle data|
|12|altitude|Numeric|m|Vehicle data|
|13|id vehicle|Numeric||Vehicle data|
|14|heart rate|Numeric|bpm|Driver's data|
|15|body temperature|Numeric|C|Driver's data|
|16|id driver|Numeric||Driver's data|
|17|current weather|Categorical||Weather data|
|18|has precipitation|Boolean||Weather data|
|19|is day time|Boolean||Weather data|
|20|temperature|Numeric|C|Weather data|
|21|wind speed|Numeric|km/h|Weather data|
|22|wind direction|Numeric||Weather data|
|23|relative humidity|Numeric|%|Weather data|
|24|visibility|Numeric|km|Weather data|
|25|uv index|Numeric||Weather data|
|26|cloud cover|Numeric||Weather data|
|27|ceiling|Numeric|m|Weather data|
|28|pressure|Numeric|mb|Weather data|
|29|precipitation|Numeric|mm|Weather data|
|30|accidents on site|Numeric|deaths|Traffic accidents|
|31|design speed|Numeric|km/h|Road geometrics characteristics|
|32|accidents time|Numeric|deaths|Road geometrics characteristics|

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

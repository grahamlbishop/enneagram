# Evaluating the Enneagram of Personality Using GloVe Vectors

This GitHub repository contains all materials related to the project "Evaluating Enneagram Types using GloVe Vectors." This project was completed as part of a graduate-level course on computational research methods.

### Project Overview

The Enneagram of Personality is a typology system that claims individuals develop dominant desires, fears, and attachment strategies corresponding to one of nine distinct "core types." While the Enneagram has gained popularity as a tool for self-understanding and personal development, its acceptance as a valid instrument of psychological assessment among academics and clinicians has been mixed.


In this project, I sought to evaluate the claim that the Enneagram's nine core types somehow exist beyond the instruments used to assess them. If individuals indeed reveal their type through the language they use, as Enneagram practitioners claim, then everyday discourse should reveal the nine types' distinctions.


### Methodology


Data: I downloaded the fifty-dimensional version of Pennington et al.'s pre-trained "GloVe" vectors, derived from data on Twitter (Pennington et al., 2014).


Word Selection: I chose ten words for each of the nine Enneagram types based on items in the Riso-Hudson Enneagram Type Indicator (RHETI) and the Wagner Enneagram Personality Style Scales (WEPSS), the two assessment scales found to have the highest reliability and validity (Riso & Hudson, 1999; Wagner, 2022). Each list of ten contained five words representing the healthier attributes of each type and five words representing the less healthy traits.


Data Visualization: I loaded the GloVe vectors into Python and used the t-distributed stochastic neighbor embedding (t-SNE) method from the scikit-learn library to reduce the 50-dimensional vectors to points on a two-dimensional space. I then visualized these points using scatterplot tools in the Matplotlib library.



### Results

An initial plot of all 90 words did not reveal the expected clustering of Enneagram types (see "Individual Word Vector Distances.png"). A subsequent plot comparing the eighteen averaged vectors of each set of five words showed that all nine averaged positive term vectors were closer to at least one of the averaged negative term vectors for the other eight types than they were to the averaged negative term vector of the corresponding type (see "Average Type Vector Distances.png").


### Conclusion

This study failed to find compelling evidence of the Enneagram's manifestation in the language used on Twitter. If Enneagram types are real (and the present study can neither support nor falsify this claim), they likely do not reveal themselves in the communications people exchange on social media.


### References


Hook, J. N., Hall, T. W., Davis, D. E., Van Tongeren, D. R., & Conner, M. (2020). The enneagram: A systematic review of the literature and directions for future research. Journal of Clinical Psychology, 77(4), 865–883. https://doi.org/10.1002/jclp.23097

Pennington, J., Socher, R., & Manning, C.D. (2014). GloVe: Global Vectors for Word Representation [Data set]. Retrieved December 3, 2022, from https://nlp.stanford.edu/projects/glove/

Riso, D. R., & Hudson, R. (1999). The Wisdom of the Enneagram: The Complete Guide to Psychological and Spiritual Growth for the Nine Personality Types. Bantam Books.

Theiler, S. (2022, July 5). "Basics of using pre-trained GloVe vectors in Python." Medium. Retrieved December 3, 2022, from https://medium.com/analytics-vidhya/basics-of-using-pre-trained-glove-vectors-in-python-d38905f356db

Wagner, J. (2022). Wagner Enneagram Personality Style Scales. Retrieved December 3, 2022, from https://wepss.com/

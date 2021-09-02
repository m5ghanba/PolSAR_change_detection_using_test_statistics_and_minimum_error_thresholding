# PolSAR_change_detection_using_test_statistics
Several complex-PolSAR-data-based test statistics and thresholding algorithms for change detection using PolSAR data

Please refer to the paper Ghanbari, M., and Akbari, V., (2018, Published). “Unsupervised Change Detection in Polarimetric SAR Data With the Hotelling-Lawley Trace Statistic and Minimum-Error Thresholding”. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing.

1. Create a .mat file including two co-registered multilook covariance data named "C1" and "C2", and a ground truth file named "GT". This file should have the name "TwoCoregisteredMultilookedCovarianceData.mat".

2. In the file GT, the change and no-change areas should be labeled as 1 and 2, respectively.

3. In the GUI file, go to "file", and, then, choose the option:

"Import time1 & time2 files for change detection". In the opened window, choose ".mat files", and press "Import and load corresponding files". The software will load the TwoCoregisteredMultilookedCovarianceData.mat file.

4. Then, in the main menu, Press "generation of scalar feature image" and then "trace statistics".

5. Upon pressing the button "Trace statistic image generation", it will generate the A^-1B, B^-1A, and the maximum image separately.

6. Finally, in the main menu, choose "Thresholding" and then "KI Thresholding". Choose the PDF and estimation methods and press the button "Thresholding". Mostly, FS distribution and, in some cases, generalized Gamma distribution do not give proper results! This is because of lacking proper estimates for the PDF parameters with the given data set. However, other distributions, especially Gamma PDF, do the job most of the time. Regarding the estimation method, choose "log cumulant estimation" which gives very good answers in cases where the other methods do not work.

7. In the "Thresholding" menu, one can choose the "test optimal" method, which gives the best change detection results in terms of error rate. I have used this supervised method to assess the thresholding algorithm in the paper.

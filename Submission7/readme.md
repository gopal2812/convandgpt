## Aim
Achieve 99.4% Test Accuracy in 15 epochs or less with an 8000 parameter or less model.

## Approach
We do three iterations starting from a baseline from the previous assignment taking steps towards the aim. The models are specified in the model.py file numbered from model1 to model3.

There are three other files model1.ipynb to model3.ipynb which are responsible for running and testing the models. They also contain the Target, Results and Analysis after each run.

The utils.py file contains the additional methods required for the model training.


Model3 RF calculation
Layer#	Input Channel (n_in)	Input RF (r_in)	Kernel size (k)	Padding (p)	Stride (s)	Input Jump (j_in)	Output Jump (j_out)	Output Channels (n_out)	Output RF (r_out)
Conv2d	 28 	1	3	0	1	1	1	 26 	3
Conv2d	 26 	3	3	0	1	1	1	 24 	5
Conv2d	 24 	5	3	0	1	1	1	 22 	7
MaxPool2d	 22 	7	2	0	2	1	2	 11 	8
Conv2d	 11 	8	1	0	1	2	2	 11 	8
Conv2d	 11 	8	3	0	1	2	2	 9 	12
Conv2d	 9 	12	3	0	1	2	2	 7 	16
Conv2d	 7 	16	3	0	1	2	2	 5 	20
Conv2d	 5 	20	1	0	1	2	2	 5 	20
![image](https://github.com/gopal2812/convandgpt/assets/39087216/7313a7ee-a126-462d-978c-7c3e1a404484)

This program is aimplemention of the conjugate gradient method, with the addition of the steepest descent 
alogorithm for the research part of assigment 2 function in parallel to resold the same 
problem provided to us in the assigment specification.

The main.py contains both the conjugate gradiet and steepest descent algoritm,
by simply running main.py, the program will procceed to perform the and resolve the function provided
'(x1 - 1)**2 + (x2 - 2)**2 +(x3 - 3)**2', using the algorithm stated, the program will operate with 
the default starting point of [0.5, 0.5, 0.5].

when running the program, 3 files will be generated:

solutions.txt				for the results of the different methods

ConjugateGradientLog.txt		detailed logs for variables at different itteration

steepestdescentlog.txt			Logs for the steepest descent itteration



I have clearly stated input position for assigment, which should allow for easy alterations by the user,
the alterations allowed within this program and their line positions are:

conjugate gradient function		line 7	

staring point				line 11
	
steepest steepestdecsent function	line 16

Due to the differnt algorithms requiring differnt data structures depending the complexity of the method
conjugate gradient function and steepest steepestdecsent function are called differntly, however the starting point
is the same since its pretty straight forward.



The steepest descent is much a much simpler method and a result it makes robust various user input




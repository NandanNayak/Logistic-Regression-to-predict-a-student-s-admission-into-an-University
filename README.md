# Logistic-Regression-to-predict-a-student's-admission-into-an-University

###Project Description

The aim of the project is determine if a student can gain admission into a university based on the scores of two exams. The prediction is done based on the decision boundary that the logistic regression learns.

###Format of the data

The input file is <a href="https://github.com/NandanNayak/Logistic-Regression-to-predict-a-student-s-admission-into-an-University/blob/master/ex2data1.txt">ex2data1.txt</a>. The first two columns contain the scores of Exam1 and Exam2 respectively. The third column contains whether the student got admission into the University or not.

###Algorithm Description

<p>
<ol>
<li>Since the features vary by orders of magnitude, the featues are normalized so that the Gradiant Descent can minimize quickly. To normalize the feature, the mean value is subtracted from the feature and then divided by standard deviation.</li>
<li>The Gradiant Descent is implemented as a second step. The cost function 'J' is parameterized by theta. In each iteration the values of theta changes based on the gradiant descent equation given below.</li>
<img src="https://github.com/NandanNayak/Linear-Regression-Model-to-predict-the-Price-of-Houses/blob/master/GradiantDescentEqn.png" align="center" />
<li>The cost function is calculated which is expected to reduce with each iteration. The formula is given as below.</li> 
<img src="https://github.com/NandanNayak/Linear-Regression-Model-to-predict-the-Price-of-Houses/blob/master/CostFunctionEqn.png" align="center" />
</ol>
</p>

###Running the code

<strong><em>python nayak_nandan_LogisticRegression.py</strong></em>

###Program Output

The program outputs the model parameters calculated from both the Gradiant Descent equation.
<a href="https://github.com/NandanNayak/Logistic-Regression-to-predict-a-student-s-admission-into-an-University/blob/master/Output.png"></a>

###Visualization and Conclusion

<img src="https://github.com/NandanNayak/Logistic-Regression-to-predict-a-student-s-admission-into-an-University/blob/master/DecisionBoundary.png" />
In the first figure, the green dots represent the students who have been admitted and the red dots represent the students who have been rejected by the University. The blue line is the decision boundary which the logistic algorithm has learnt from the dataset. Inorder to predict the admission of future students, they are plotted on the graph and those that only lie on the right side of the decision boundary get admitted.

The second figure shows the variation of the cost function with the no. of iterations of the Gradient Descent.

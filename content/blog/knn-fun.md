+++
date = '2021-01-19T06:09:46-04:00'
title = 'Fun Intro to K-NN Algorithm'
#
# description is optional
#
# description = "An optional description for SEO. If not provided, an automatically created summary will be used."

tags = []
+++


## Machine Learning Nuggets: A fun Introduction to K-NN Algorithm

A girl from an orthodox family loved a boy while in college.

She informed her parents about this relation and expressed interest that she would like to marry him.

She shared all the details about the boy to her parents that she knew.

After some usual family drama, parents agreed to the wedding but with one condition that only if the boy passes background check.

Now, her parents would like to find more about the boy and take decision of go or no-go.

The parents arrived at boy’s place, and started inquiring about the boy with nearby residents.
They were able to speak with 3-families which are very close to boy’s house.

### Here are the responses look like

---
Left-Side Resident: Endorsed the boy and wished good luck [positive]

Right-Side Resident: They didn’t endorse [negative]

Other Resident: Shared all good things about the boy [positive]

Out of these three evidences, only 1-negative feedback which results to a final positive overall as the majority of votes are positive.

The parents are now peacefully agreed, and the girl is happy :-)

---
That’s K-NN algorithm applied by parents.
Here, k=3, three nearest neighbors and
We have 2-classes (Go/No-Go) — binary classification
Feedback=Vote: Weight =1

Hope the above story was fun to read and intuitive example of K-Nearest Neighbors!

---
Now, let us learn more about how K-NN works.

The k-nearest neighbors (KNN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems.

```bash
Classification: When the target variable is categorical (ex: Yes/No)

Regression: When target variable is numerical (ex: House Prices)
```

### High level Steps of K-NN

Find k-nearest points (uses the euclidean distance) of a point on which you want to make a prediction
Voting or Average of Multiple Neighbors
For regression (numerical value) problems — average (i.e. mean) is used.

For classification (categorical value) problems — we take majority of voting, that is mode, value that is observed frequently in a set.